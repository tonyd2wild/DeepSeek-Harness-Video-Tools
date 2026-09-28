# DeepSeek-Harness-Video-Tools

> ⚠️ **Unofficial community project.** Not affiliated with or endorsed by DeepSeek or MiniMax. Community tooling for the [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (`dsh`) agent harness.
>
> 🔒 **Security note:** this tool has no authentication of its own and talks to your ComfyUI endpoint in plain HTTP. Keep both on `127.0.0.1` or inside a trusted network (Tailscale/WireGuard), and never expose them directly to the internet.

> **Requires dsh 0.2.0-rc.1 or newer** (web UI and desktop app). On dsh 0.1.x, use the [`dsh-0.1` tag](https://github.com/tonyd2wild/DeepSeek-Harness-Video-Tools/tree/dsh-0.1).

Model-facing **`generate_video`** and **`check_video`** tools for DeepSeek Harness, backed by **your own local ComfyUI + MiniMax H3 deployment** — the model makes the audio too. No cloud, no API key, no per-minute billing.

```
generate_video("an empty motel corridor at night...", seconds: 5)
  → { job_id: "vid_20260921..._x1eu", eta: "about 6 min" }     # returns IMMEDIATELY
check_video("vid_20260921..._x1eu")
  → "/path/to/DSHSMOKE_00001_.mp4 — video done (0.4 MB, 90 frames)"  # when finished
```

## Why async is mandatory

A video is **6–12 minutes**; a 15-second clip is **~2.5 hours**; and the cost is badly **super-linear** in frame count (5s→8s doubles; 8s→15s multiplies by ~25 — that is how spatio-temporal attention scales). A blocking tool call would hit a timeout somewhere while the GPU happily finishes a job nobody collects. So:

- **`generate_video`** validates the request, picks a free lane, dispatches, and returns `{ job_id, eta }` **immediately** — it never blocks.
- **`check_video`** polls the job; returns the finished mp4's absolute path when done, or the time remaining if still rendering.

Every response carries the ETA so the agent can keep the human informed instead of going silent.

## What it gives the agent

| knob | values | notes |
|---|---|---|
| `seconds` | snaps **DOWN** to the **17n+5 grid** at 24 fps | 124f≈5.2s … 362f≈15.1s; 362 is the trained maximum — anything longer needs split-and-stitch (refused in v1) |
| `quality` | `"full"` (default) / `"fast"` | full = 20 steps on the base checkpoint; fast = 8 steps on the fused-turbo checkpoint (~2-3× quicker, but it can fail character swaps on hard shots) |
| `style` | `"none"` (default) / `"80s_horror"` | the only style LoRA in this fleet; its trigger word is auto-inserted |
| `first_frame` | absolute image path | anchors the opening frame |
| — | **no cfg knob** | H3 runs the no-CFG path (`BasicGuider` → `SamplerCustomAdvanced`); there is nothing to tune |
| — | **no resolution knob** | 960×544 is the only properly tested resolution; cost is super-linear in pixels (2.14× time for 1.76× pixels measured) |

**The SigmaShift rule** (the trap most people get wrong): `MiniMaxH3SigmaShift` (12.0/3.0) applies **only** to turbo/step-distilled runs — it exists to re-match a distilled noise schedule. A style LoRA must NOT get it, and a no-LoRA run must NOT get it. The tool applies this rule for you.

## Prompt format (H3 wants structure, not prose)

```
integrated_multimodal_description: <style line>. <the scene, in detail>.

[Shot 1] 0.0s to <N>s, one continuous shot. <what the camera does, what happens and when>

Constraints: <what must NOT appear; end with "every frame sharp and detailed">

overall_soundscape: <H3 generates the audio — describe it>

non_diegetic_music: N/A.
```

Measured rules that matter:
- **Never ask for mood words.** grain, halation, glow, haze, smoke, soft, dreamy all tell the model to render soft, and it obeys: sharpness 33–100 with them, 300+ without.
- **Give the camera a constant velocity** and say "no speed ramps" — otherwise slow push-ins become fast dollies.
- **Name famous characters, describe generic ones.** Naming pulls a stronger result for characters the model knows; describing fights it for them.
- **Never accept a sharpness metric as proof of correctness.** The 8-step turbo scored 3× higher sharpness on a *broken* crowd swap than the correct base render. Pull frames and look.

## The traps this tool already handles

- **The outputs key is `images`, not `video`.** A finished H3 job reports `outputs[node]["images"] = [{filename, subfolder, type}]` — a poller that looks for `"video"` waits forever while the file sits finished on disk.
- **A failed LoRA load is silent.** The lane log must say `208 patches attached`; `0 patches` means it rendered plain. (The tool's style path uses the verified `1980s_horror_h3_175.safetensors`.)
- **Lane B is shared.** The tool checks `/queue` before dispatch and refuses with a clear message rather than queueing behind a two-hour job.
- **VRAM co-tenancy:** an image lane and a video lane cannot both be resident on one 24 GB card (H3 ≈ 16.5 GB + Qwen-Image ≈ 17 GB). The tool frees the sibling image lane (`POST /free`) before dispatching.

## Requirements

- **dsh 0.2.0-rc.1 or newer**, in the web UI (profile `web`) and/or DeepSeek's desktop app (profile `desktop`). npm's `latest` tag still points at 0.1.x at the time of writing, so install with `npm i -g @deepseek-ai/dsh@next`
- **Using dsh 0.1.x?** Use the [`dsh-0.1` tag](https://github.com/tonyd2wild/DeepSeek-Harness-Video-Tools/tree/dsh-0.1) of this repo; its install steps (`agent.cordis.yml` preset folders) do not apply to 0.2
- **Node** `^22.19.0 || >=24.0.0` (what dsh 0.2 needs) and **pnpm** (`dsh plugin` shells out to it)
- **ComfyUI** with a MiniMax H3 checkpoint, reachable over HTTP

## Install

On dsh 0.2 all configuration lives in `$DSH_HOME/profiles/<profile>/cordis.patch.yml`
(default `$DSH_HOME` is `~/.dsh`). Repeat steps 3 to 5 for each profile you use:
`web`, `desktop`, or both.

```bash
# 1. clone next to your other plugins
git clone https://github.com/tonyd2wild/DeepSeek-Harness-Video-Tools.git ~/.dsh/plugins/video-tools
cd ~/.dsh/plugins/video-tools

# 2. link the harness's OWN @deepseek-ai/dsh-tools, so the tools are built with the exact copy your dsh runs
npm pkg set "dependencies.@deepseek-ai/dsh-tools=link:$(npm root -g)/@deepseek-ai/dsh/node_modules/@deepseek-ai/dsh-tools"
pnpm install --ignore-scripts

# 3. install it into the profile (never into the dsh install's node_modules)
dsh plugin --profile web     add link:/absolute/path/to/.dsh/plugins/video-tools
dsh plugin --profile desktop add link:/absolute/path/to/.dsh/plugins/video-tools
```

**4. Add the tools to your agent preset** in that profile's `cordis.patch.yml`.
The row is referenced by the package's name, `DeepSeek-Harness-Video-Tools`:

```yaml
          - id: tool-video-gen
            name: 'DeepSeek-Harness-Video-Tools'
            config:
              lanes:      ["http://127.0.0.1:8188", "http://127.0.0.1:8189"]
              imageLanes: ["http://127.0.0.1:8190", "http://127.0.0.1:8191"]
              outDir:     /absolute/writeable/output/dir
```

On 0.2 a preset is an `@deepseek-ai/dsh-agent-preset` row. You cannot append a
tool to the shipped `standard` preset (a patch replaces a row's whole config), so
the tool row goes into a preset row of your own that copies the shipped plugin
list, made default through the `agent-preset-registry` row. If another community
tool already gave you such a row, append `tool-video-gen` to it. Otherwise start
from the hub's
[one-preset-for-all-tools example](https://github.com/tonyd2wild/DeepSeek-Harness-Tools/blob/main/examples/cordis.patch.yml)
and uncomment the Video Tools row.

**5. Restart dsh** (web UI: stop and re-run `dsh web`; desktop app: quit fully
and reopen), then create a **new** session: presets mount lazily, and a running
session keeps its old tool catalog.

## Configuration

All fleet specifics live in the tool row's `config:`, nothing about our hardware
is hardcoded ([config.example.json](config.example.json) shows the keys):

| key | default | meaning |
|---|---|---|
| `lanes` | `["http://127.0.0.1:8188","http://127.0.0.1:8189"]` | ComfyUI video endpoints, preferred first |
| `imageLanes` | (sibling of each lane) | the image endpoints sharing each video lane's card, freed before dispatch |
| `outDir` | `./output` | where finished mp4s are written |
| `maxFrames` | `362` | the trained ceiling; over this needs split-and-stitch |

## The recipe (measured on the box)

`fl2va` unet + NVFP4-AWQ text encoder (`type: minimax`) + `MiniMaxH3ImageToVideo` (no image wired = pure t2v), `res_multistep`/`simple`, denoise 1.0. Checkpoints: `minimax_h3_ref2va_pruned_int8_convrot.safetensors` (base/full), `minimax_h3_ref2va_pruned_turbo_int8_convrot.safetensors` (fused turbo/fast), `minimax_h3_video_vae_fp16.safetensors`, `minimax_h3_audio_vae_fp32.safetensors`. Style LoRA: `1980s_horror_h3_175.safetensors` (trigger `80s_horror`).

**Measured:** 90 frames / 8 steps / 960×544 = **~3 min**; 124f/20 steps ≈ 6 min; 192f ≈ 12 min; 362f ≈ 2.5 h — all on one RTX 3090.

## Sibling projects

- [DeepSeek-Harness-Image-Tools](https://github.com/tonyd2wild/DeepSeek-Harness-Image-Tools) — `generate_image` (ComfyUI + Qwen-Image)
- [DeepSeek-Harness-Tools](https://github.com/tonyd2wild/DeepSeek-Harness-Tools) — the hub index of all dsh community tools
- [DeepSeek-Harness-Vision-Tools](https://github.com/tonyd2wild/DeepSeek-Harness-Vision-Tools) — `analyze_image` (give dsh eyes)
- [DeepSeek-Harness-Web-Tools](https://github.com/tonyd2wild/DeepSeek-Harness-Web-Tools) — keyless `web_search` / `web_fetch`
- [DeepSeek-Harness-Browser](https://github.com/tonyd2wild/DeepSeek-Harness-Browser) — the in-app browser pane

## Contributing

Issues and PRs welcome. Keep the "⚠️ Unofficial community project" banner, keep endpoints in config, and remember the design constraint: **tools return text — a file path — never a video payload.**

## License

MIT — see [LICENSE](LICENSE).
