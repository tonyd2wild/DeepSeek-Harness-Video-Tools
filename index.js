/**
 * dsh-plugin-video-tools: `generate_video` / `check_video` model-facing tools
 * for DeepSeek Harness, backed by a LOCAL ComfyUI + MiniMax H3 deployment (no
 * API key, no cloud).
 *
 * Design (measured constraints, see README):
 *  - A video is 6-12 MINUTES (a long one ~2.5 h) and the cost is super-linear
 *    in frame count. So the async two-call shape is MANDATORY:
 *    generate_video returns { job_id, eta } immediately and NEVER blocks;
 *    check_video polls.
 *  - The tool returns TEXT — a file path plus a one-line confirmation — and
 *    never a video payload into the model's context. The agent opens the mp4
 *    with an in-app preview tool.
 *  - `length` must sit on the 17n+5 grid; 362 frames / 15.1 s is the trained
 *    maximum. Seconds are exposed; frames snap DOWN and the snap is reported.
 *  - The ComfyUI outputs key is `images` even for video (a finished job
 *    reports outputs[node]["images"]). Polling for "video" waits forever while
 *    the file sits finished on disk.
 *  - Lanes: prefer lane A; B is commonly shared. An image lane and a video
 *    lane cannot both be resident on one 24 GB card, so the sibling image lane
 *    is freed via POST /free before dispatching.
 *  - SigmaShift (12.0/3.0) applies ONLY to turbo/step-distilled runs. A style
 *    LoRA must not get it, and neither must a no-LoRA run.
 *
 * All fleet specifics live in the plugin config (see config.example.json).
 *
 * @module dsh-plugin-video-tools
 */

import { defineTool } from '@deepseek-ai/dsh-tools'
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import path from 'node:path'

/** Fleet-agnostic defaults; every one is overridable via the plugin's config. */
const DEFAULTS = {
  lanes: ['http://127.0.0.1:8188', 'http://127.0.0.1:8189'],
  imageLanes: ['http://127.0.0.1:8190', 'http://127.0.0.1:8191'],
  outDir: './output',
  maxFrames: 362,
  fps: 24,
  files: {
    unetFull: 'minimax_h3_ref2va_pruned_int8_convrot.safetensors',
    unetTurbo: 'minimax_h3_ref2va_pruned_turbo_int8_convrot.safetensors',
    clip: 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',
    videoVae: 'minimax_h3_video_vae_fp16.safetensors',
    audioVae: 'minimax_h3_audio_vae_fp32.safetensors',
  },
  styles: {
    none: null,
    '80s_horror': { file: '1980s_horror_h3_175.safetensors', trigger: '80s_horror' },
  },
}

/** Per-call HTTP timeouts. */
const REQUEST_TIMEOUT_MS = 30_000
/** Video jobs are minutes long; poll gently. */
const POLL_INTERVAL_MS = 15_000

/** The checkpoint + steps per quality tier (all values measured, see README). */
const QUALITY = {
  fast: { unetKey: 'unetTurbo', steps: 8, sigmaShift: true },
  full: { unetKey: 'unetFull', steps: 20, sigmaShift: false },
}

/** SigmaShift values for turbo runs (the node's working defaults). */
const TURBO_SHIFT = { video: 12.0, audio: 3.0 }

/** fetch with an abort deadline. */
async function fetchJson(url, { method = 'GET', body, timeoutMs } = {}) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs ?? REQUEST_TIMEOUT_MS)
  try {
    const response = await fetch(url, {
      method,
      headers: body !== undefined ? { 'content-type': 'application/json' } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    })
    if (!response.ok) {
      let detail = `HTTP ${response.status}`
      try {
        const p = await response.json()
        if (p?.error) detail = typeof p.error === 'string' ? p.error : JSON.stringify(p.error)
      } catch { /* keep status */ }
      throw new Error(`ComfyUI ${method} ${url} failed: ${detail}`)
    }
    return await response.json()
  } catch (error) {
    if (error?.name === 'AbortError') throw new Error(`ComfyUI request timed out: ${method} ${url}`)
    if (error?.cause?.code === 'ECONNREFUSED') throw new Error(`ComfyUI lane unreachable: ${url} (connection refused)`)
    throw error
  } finally {
    clearTimeout(timer)
  }
}

/** Is this lane free (nothing running, nothing pending)? Unreachable = busy. */
async function laneBusy(lane) {
  try {
    const q = await fetchJson(`${lane}/queue`, { timeoutMs: 8_000 })
    return (q?.queue_running?.length ?? 0) + (q?.queue_pending?.length ?? 0) > 0
  } catch {
    return true
  }
}

/** Pick the video lane to dispatch on, preferring the first. */
async function pickLane(lanes) {
  for (const lane of lanes) {
    if (!(await laneBusy(lane))) return { lane, busyAll: false }
  }
  return { lane: null, busyAll: true }
}

/** Free an image lane's resident weights so the video lane fits on the card. */
async function freeImageLane(imageLane) {
  try {
    await fetchJson(`${imageLane}/free`, { method: 'POST', body: { unload_models: true, free_memory: true }, timeoutMs: 20_000 })
    return true
  } catch {
    return false
  }
}

/** Seconds -> largest 17n+5 frame count NOT above seconds*24, snapped DOWN. */
function snapFrames(seconds, fps, maxFrames) {
  const raw = Math.max(1, Math.floor(Number(seconds) || 0) * fps)
  const frames = Math.max(5, Math.floor((raw - 5) / 17) * 17 + 5)
  return { frames: Math.min(frames, maxFrames), snappedFrom: raw }
}

/** Rough ETA (s) from the measured table; piecewise-linear on frames. */
function etaSeconds(frames) {
  const pts = [[90, 174], [124, 360], [192, 720], [362, 9000]]
  if (frames <= pts[0][0]) return Math.round(174 * frames / 90)
  for (let i = 1; i < pts.length; i++) {
    const [f0, t0] = pts[i - 1]
    const [f1, t1] = pts[i]
    if (frames <= f1) return Math.round(t0 + (t1 - t0) * (frames - f0) / (f1 - f0))
  }
  return 9000
}

/** Format seconds as "~1.5 min" / "~6 min" / "~2.5 h". */
function fmtDuration(s) {
  if (s < 90) return 'about ' + (Math.round(s / 6) / 10).toFixed(1) + ' min'
  if (s < 3600) return 'about ' + Math.round(s / 60) + ' min'
  return 'about ' + (Math.round(s / 360) / 10).toFixed(1) + ' h'
}


/** The H3 text-to-video graph (fl2va recipe): optional style LoRA, optional first frame. */
function buildGraph({ prompt, negative, frames, width, height, seed, quality, styleLora, files, fps, firstFrameB64 }) {
  const q = QUALITY[quality] ?? QUALITY.full
  const g = {
    // checkpoint per tier (fast = fused turbo, full = base)
    '1': { class_type: 'UNETLoader', inputs: { unet_name: files[q.unetKey], weight_dtype: 'default' } },
    '2': { class_type: 'CLIPLoader', inputs: { clip_name: files.clip, type: 'minimax', device: 'default' } },
    // SigmaShift ONLY for a turbo/step-distilled run; the graph wires node 3
    // unconditionally and gives it identity shift when not needed — simplest
    // correct shape on lanes without a Reroute node.
    '3': { class_type: 'MiniMaxH3SigmaShift', inputs: { model: ['1', 0], shift_video: q.sigmaShift ? TURBO_SHIFT.video : 1.0, shift_audio: q.sigmaShift ? TURBO_SHIFT.audio : 1.0 } },
    // AV latent on the 17n+5 grid
    '4': { class_type: 'EmptyMiniMaxH3LatentAV', inputs: { width, height, length: frames, batch_size: 1 } },
    // carries the structured prompt; no image wired => pure text-to-video
    '5': { class_type: 'MiniMaxH3ImageToVideo', inputs: { clip: ['2', 0], vae: ['8', 0], prompt, width, height, length: frames } },
    '6': { class_type: 'CLIPTextEncode', inputs: { clip: ['2', 0], text: negative } },
    '7': {
      class_type: 'KSampler',
      inputs: {
        model: ['3', 0], positive: ['5', 0], negative: ['6', 0], latent_image: ['5', 1],
        seed, steps: q.steps, cfg: 1.0, sampler_name: 'res_multistep', scheduler: 'simple', denoise: 1.0,
      },
    },
    '8': { class_type: 'VAELoader', inputs: { vae_name: files.videoVae } },
    '9': { class_type: 'VAELoader', inputs: { vae_name: files.audioVae } },
    // split decode: frames from the video VAE, audio from the audio VAE
    '10': { class_type: 'VAEDecode', inputs: { samples: ['7', 0], vae: ['8', 0] } },
    '11': { class_type: 'VAEDecodeAudio', inputs: { samples: ['7', 0], vae: ['9', 0] } },
    // mux frames + generated audio, then save mp4
    '12': { class_type: 'CreateVideo', inputs: { images: ['10', 0], fps, audio: ['11', 0] } },
    '13': { class_type: 'SaveVideo', inputs: { video: ['12', 0], filename_prefix: 'video/dsh_gen', format: 'mp4', codec: 'h264' } },
  }
  // Style LoRA (model-only, after the shift).
  if (styleLora) {
    g['14'] = { class_type: 'LoraLoaderModelOnly', inputs: { model: ['3', 0], lora_name: styleLora.file, strength_model: 1.0 } }
    g['7'].inputs.model = ['14', 0]
  }
  // Optional opening-frame anchor.
  if (firstFrameB64) {
    g['20'] = { class_type: 'LoadImageFromBase64', inputs: { image: firstFrameB64 } }
    g['5'].inputs.first_frame = ['20', 0]
  }
  return g
}

/** A collision-proof job id. */
function newJobId() {
  const stamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14)
  return `vid_${stamp}_${Math.random().toString(36).slice(2, 8)}`
}

/** In-memory job registry: job_id -> state. A restart forgets, and says so. */
const jobs = new Map()

/**
 * Register the two tools. Resolved config closes over them, stable for the
 * process lifetime.
 *
 * @param {import('@deepseek-ai/cordis').Context} ctx
 * @param {{lanes?: string[], imageLanes?: string[], outDir?: string, maxFrames?: number, styles?: object, files?: object, enabled?: boolean}} [config]
 */
export function apply(ctx, config) {
  const lanes = Array.isArray(config?.lanes) && config.lanes.length > 0 ? config.lanes : DEFAULTS.lanes
  const siblingImageLanes = Array.isArray(config?.imageLanes) && config.imageLanes.length === lanes.length ? config.imageLanes : DEFAULTS.imageLanes
  const outDir = typeof config?.outDir === 'string' && config.outDir.trim().length > 0 ? config.outDir : DEFAULTS.outDir
  const maxFrames = Number.isInteger(config?.maxFrames) && config.maxFrames > 0 ? config.maxFrames : DEFAULTS.maxFrames
  const fps = (typeof config?.fps === 'number' && config.fps > 0) ? config.fps : DEFAULTS.fps
  const files = (config?.files && typeof config.files === 'object') ? { ...DEFAULTS.files, ...config.files } : DEFAULTS.files
  const styles = (config?.styles && typeof config.styles === 'object') ? { none: null, ...config.styles } : DEFAULTS.styles
  const styleNames = Object.keys(styles).filter(k => k !== 'none').map(s => `"${s}"`).join(', ')

  ctx.tools.register(defineTool({
    name: 'generate_video',
    description:
      'Start a LOCAL text-to-video generation (MiniMax H3 via ComfyUI; the model makes the audio too). '
      + 'RETURNS IMMEDIATELY with a job_id and an ETA — it does NOT wait; poll with check_video(job_id). '
      + `seconds is the clip length; it snaps DOWN to the 17n+5 frame grid at ${fps}fps and the response says what it snapped to. Max ${maxFrames} frames (~15.1s): longer clips need splitting and stitching (not supported yet; the tool refuses). `
      + 'quality: "full" (20 steps on the base checkpoint — the production setting, default) or "fast" (8 steps on the fused-turbo checkpoint, its native schedule; ~2-3x quicker but it can fail character swaps on hard shots). '
      + `style: "none" (default)${styleNames ? ` or ${styleNames}` : ''} — the style restyles the whole render (it is not scene content). `
      + 'Videos take 6-12 minutes (up to ~2.5h at 15s); ALWAYS tell the user the ETA from the response and that they can ask "check the video" later. '
      + 'Prompt format: H3 wants the structured form — "integrated_multimodal_description: <style>. <scene>. [Shot 1] 0.0s to Ns, one continuous shot. <camera action>. Constraints: <what must not appear; end with \'every frame sharp and detailed\'>. overall_soundscape: <sound>. non_diegetic_music: N/A." '
      + 'NEVER use mood/softness words (grain, halation, glow, haze, smoke, soft, dreamy) — they measurably soften the render. Give the camera a constant velocity and say "no speed ramps". '
      + 'Name famous characters rather than describing them; describe generic ones. '
      + 'first_frame: optional absolute path to an image anchoring the opening frame. There is no cfg knob (H3 runs the no-CFG path) and 960x544 is the only tested resolution.',
    parameters: {
      prompt: { type: 'string', required: true, description: 'Structured H3 prompt (see description). Multi-line is fine.' },
      seconds: { type: 'string', description: 'Desired clip length in seconds. Snaps down to the 17n+5 grid; 5-15 recommended. Defaults to 5.' },
      quality: { type: 'string', description: '"full" (20 steps, base checkpoint, best quality — default) or "fast" (8 steps, turbo checkpoint, ~2-3x quicker).' },
      style: { type: 'string', description: `"none" (default)${styleNames ? ` or ${styleNames}` : ''}.` },
      first_frame: { type: 'string', description: 'Optional absolute path to an image to anchor the opening frame.' },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          description: { type: 'string', required: true },
          jobId: { type: 'string', required: true },
        },
      },
      render: (_args, value) => [{ type: 'text', text: value.description }],
    },
    timeoutMs: REQUEST_TIMEOUT_MS * 2,
    isConcurrencySafe: () => false,
    async execute(args) {
      const prompt = typeof args.prompt === 'string' ? args.prompt.trim() : ''
      if (prompt.length === 0) throw new Error('prompt must be a non-empty string')

      // Seconds -> frames on the grid, snapped DOWN, and the snap reported.
      const secondsNum = Number(args.seconds) || 5
      const { frames, snappedFrom } = snapFrames(secondsNum, fps, maxFrames)
      if (Number(args.seconds) > 15.08 && snappedFrom > maxFrames) {
        throw new Error(`Requested ${secondsNum}s exceeds the trained maximum of 15.08s (${maxFrames} frames). `
          + 'Clips must be split and stitched for longer runs; for now ask for at most 15 seconds.')
      }
      const eta = etaSeconds(frames)
      const etaText = fmtDuration(eta)

      const { lane, busyAll } = await pickLane(lanes)
      if (busyAll) {
        throw new Error(`All video lanes (${lanes.join(', ')}) are busy with long renders right now. Try again in a few minutes — do not queue behind them.`)
      }

      // The card an image lane shares cannot hold both models: free it first.
      const freed = await freeImageLane(siblingImageLanes[lanes.indexOf(lane)] ?? siblingImageLanes[0])

      let firstFrameB64
      if (typeof args.first_frame === 'string' && args.first_frame.trim().length > 0) {
        firstFrameB64 = (await readFile(args.first_frame.trim())).toString('base64')
      }

      const quality = args.quality === 'fast' ? 'fast' : 'full'
      const style = typeof args.style === 'string' && styles[args.style] ? args.style : 'none'
      const styleLora = styles[style]

      // A style trigger note belongs in the prompt line: insert if absent.
      const finalPrompt = (styleLora?.trigger && !prompt.includes(styleLora.trigger))
        ? `${prompt}${prompt.endsWith('.') ? '' : '.'} Style: ${styleLora.trigger}.`
        : prompt

      const seed = (Date.now() ^ Math.floor(Math.random() * 0xffffffff)) >>> 0
      const graph = buildGraph({
        prompt: finalPrompt,
        negative: 'low quality, blurry, distorted, watermark, on-screen text, people, soft focus, grain, halation, glow, haze, dreamy',
        frames, width: 960, height: 544, seed, quality, styleLora, files, fps, firstFrameB64,
      })
      const dispatched = await fetchJson(`${lane}/prompt`, { method: 'POST', body: { prompt: graph, client_id: 'dsh-plugin-video-tools' }, timeoutMs: REQUEST_TIMEOUT_MS })
      if (!dispatched?.prompt_id) throw new Error(`ComfyUI accepted the prompt but returned no prompt_id: ${JSON.stringify(dispatched).slice(0, 200)}`)

      const jobId = newJobId()
      jobs.set(jobId, { promptId: dispatched.prompt_id, lane, frames, startedAt: Date.now(), etaS: eta })

      const snapNote = snappedFrom !== frames ? ` (snapped down from ${snappedFrom} raw frames)` : ''
      const tierNote = quality === 'fast' ? ', quality fast (8-step turbo)' : ', quality full (20-step base)'
      const freeNote = freed ? '' : ' (could not free the sibling image lane; watch for OOM)'
      return {
        jobId,
        description: `Video job started: job_id ${jobId} — ${frames} frames (~${(frames / fps).toFixed(2)}s at ${fps}fps)${snapNote}, 960x544${tierNote}`
          + (style !== 'none' ? `, style ${style}` : '')
          + `. ETA ${etaText}. Poll with check_video("${jobId}"). Tell the user the ETA and that they can ask you to check on it.${freeNote}`,
      }
    },
    presentCall: (args) => ({ card: 'generic', title: `Generate video${args?.seconds ? ` (${args.seconds}s)` : ''}`, kind: 'video-gen', rawInput: args?.prompt }),
    presentResult: (_args, result) => ({ card: 'generic', title: 'Video job started', kind: 'video-gen', output: result.description }),
  }))

  ctx.tools.register(defineTool({
    name: 'check_video',
    description:
      'Check a video generation job started with generate_video. Pass the job_id it returned. '
      + 'Returns the finished mp4 file path when done (open it in the preview pane to show the user), or time remaining / still-running status. '
      + 'Poll gently — a video takes minutes; if it is still running, tell the user the ETA rather than polling in a tight loop.',
    parameters: {
      job_id: { type: 'string', required: true, description: 'The job_id returned by generate_video.' },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          description: { type: 'string', required: true },
        },
      },
      render: (_args, value) => [{ type: 'text', text: value.description }],
    },
    timeoutMs: 120_000,
    isConcurrencySafe: () => true,
    async execute(args) {
      const jobId = typeof args.job_id === 'string' ? args.job_id.trim() : ''
      const job = jobs.get(jobId)
      if (!job) throw new Error(`unknown job_id "${jobId}". Jobs live in the process memory of the session that started them; `
        + 'if the harness restarted, start the video again. Known jobs: ' + ([...jobs.keys()].join(', ') || '(none)'))

      // Finished outputs use the key "images" even for video (a documented trap).
      let history
      try {
        history = await fetchJson(`${job.lane}/history/${job.promptId}`, { timeoutMs: 15_000 })
      } catch (error) {
        return { description: `Could not reach the render lane while checking job ${jobId}: ${error.message}. It may still be rendering; try again shortly.` }
      }
      const entry = history?.[job.promptId]
      const outputs = entry?.outputs
      if (outputs) {
        for (const nodeOutput of Object.values(outputs)) {
          const first = nodeOutput?.images?.[0]
          if (first?.filename) {
            const params = new URLSearchParams({ filename: first.filename, subfolder: first.subfolder ?? '', type: first.type ?? 'output' })
            const controller = new AbortController()
            const timer = setTimeout(() => controller.abort(), 60_000)
            try {
              const response = await fetch(`${job.lane}/view?${params}`, { signal: controller.signal })
              if (!response.ok) throw new Error(`HTTP ${response.status}`)
              const bytes = Buffer.from(await response.arrayBuffer())
              await mkdir(outDir, { recursive: true })
              const outPath = path.join(outDir, first.filename)
              await writeFile(outPath, bytes)
              jobs.delete(jobId)
              return { description: `${outPath} — video done (${(Math.round(bytes.length / (1024 * 1024) * 10) / 10)} MB, ${job.frames} frames). Open it in the preview pane to show the user.` }
            } catch (error) {
              return { description: `Render finished but the download failed: ${error.message}. The file is "${first.filename}" (subfolder "${first.subfolder}") on ${job.lane}; retry shortly.` }
            } finally {
              clearTimeout(timer)
            }
          }
        }
      }
      if (entry?.status?.status_str === 'error') {
        jobs.delete(jobId)
        throw new Error(`Video job ${jobId} failed on the render lane. Start it again (possibly at a lower frame count).`)
      }
      const elapsed = Math.round((Date.now() - job.startedAt) / 1000)
      const remaining = Math.max(0, job.etaS - elapsed)
      return { description: `Job ${jobId} is still rendering. About ${fmtDuration(remaining)} to go (started ${Math.round(elapsed / 60)} min ago). Tell the user; do not poll again for at least a minute or two.` }
    },
    presentCall: (args) => ({ card: 'generic', title: 'Check video job', kind: 'video-gen', rawInput: args?.job_id }),
    presentResult: (_args, result) => ({ card: 'generic', title: 'Video check', kind: 'video-gen', output: result.description }),
  }))
}
