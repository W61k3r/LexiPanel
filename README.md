# LexiPanel

**Most tools run local AI. LexiPanel fits it: measuring your exact hardware and real workload,
testing changes in idle windows, verifying each against the live requests that follow, and
rolling back regressions on its own. One self-hosted control plane for llama.cpp, image, audio
and NPU engines, with GPU tuning, guard rails and MCP control.**

Chat and coding LLMs with [llama.cpp](https://github.com/ggml-org/llama.cpp), images with
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), speech / music / sound
with [audio.cpp](https://github.com/0xShug0/audio.cpp), GGUF chat with
[Camelid](https://github.com/timtoole02/Camelid), and ONNX models on the CPU or an NPU (AMD
Ryzen AI, Intel, Qualcomm) with [ONNX Runtime](https://onnxruntime.ai/), all from one browser
tab. **Version 1.0.0**: what is in it, and what is verified, in [CHANGELOG.md](CHANGELOG.md).

![python](https://img.shields.io/badge/python-3.12%2B%20stdlib%20only-3776AB)
![engines](https://img.shields.io/badge/engines-llama.cpp%20%C2%B7%20sd.cpp%20%C2%B7%20audio.cpp%20%C2%B7%20Camelid%20%C2%B7%20ONNX%20Runtime-6f42c1)
![gpu](https://img.shields.io/badge/GPU%20%2F%20NPU-Vulkan%20%C2%B7%20ROCm%20%C2%B7%20CUDA%20%C2%B7%20Ryzen%20AI%20%C2%B7%20OpenVINO%20%C2%B7%20QNN%20%C2%B7%20CPU-d29922)
![deps](https://img.shields.io/badge/pip%20deps-none-3fb950)
![version](https://img.shields.io/badge/version-1.0.0-blue)

It grew out of running a 27B model at 131k–262k context on a single Radeon 7900 XTX, and
almost every feature exists because something broke without it. It is a **stdlib-only Python
server with zero pip dependencies**, a single-page UI, and systemd units — no Docker, no
database, no build step.

> **Heads-up:** this is a working copy of one real machine's panel, cleaned for sharing. It
> assumes a user called `admin` with everything under `/home/admin`. Read
> [Adapting it to your box](#adapting-it-to-your-box) before installing.

---

## Contents

- [The idea](#the-idea)
- [What it does](#what-it-does)
- [Tour of the tabs](#tour-of-the-tabs)
- [Engines](#engines)
- [Architecture](#architecture)
- [Install](#install)
- [Configuration files](#configuration-files)
- [Parameters](#parameters) → full reference in **[PARAMETERS.md](PARAMETERS.md)**
- [HTTP API](#http-api) → full route list in **[API.md](API.md)**
- [Safety rails](#safety-rails-learned-the-hard-way)
- [Security](#security)
- [Adapting it to your box](#adapting-it-to-your-box)
- [Workload and auto-fit](#workload-and-auto-fit)
- [GPU Tuning](#gpu-tuning)
- [MCP server](#mcp-server)
- [GG: Graph Gauntlet](#gg-graph-gauntlet)
- [Fit: hardware-fitted requants](#fit-hardware-fitted-requants)
- [macOS (experimental)](#macos-experimental)
- [ONNX Runtime: NPUs](#onnx-runtime-npus-amd-intel-qualcomm)
- [Access: single-user or multi-user](#access-single-user-or-multi-user)
- [Gateway and quotas](#gateway-and-quotas)
- [Fleet: many boxes, one primary](#fleet-many-boxes-one-primary)
- [Checks before an upload](#checks-before-an-upload)
- [File layout](#file-layout)
- [Changelog](CHANGELOG.md)
- [Credits](#credits)

---

## The idea

**Fit AI to the machine; don't just run AI on it.** Most local-AI tools answer *"how do I run
this model?"*. LexiPanel asks *"on this exact machine, with this model and this workload, which
configuration actually works best, and how do we prove it?"*, and answers with measurements
rather than rules of thumb:

| Step | Where it happens |
|---|---|
| **Discover** | GPU and GPU Tuning tabs (cards, clocks, thermals, power, VBIOS), Power options (equipment), the GGUF reader behind the memory estimator |
| **Fit** | VRAM/RAM estimator and RAM budget, launch-plan preview, Fit tab requants planned per tensor for your cards |
| **Optimize** | Optimize tab, decode-vs-depth curve, GPU Tuning benchmark |
| **Validate** | coding and agent suites, refusal check, Diagnostics, apply-and-test with automatic revert |
| **Operate** | instances, fallback tiers, start guard, crash forensics, power profiles |
| **Learn** | statistics and the live decode chart judged against the measured curve, stability per power profile, benchmark history |
| **Adapt** | Workload tab: what real requests look like (depth, length, concurrency, idle hours) and where the settings do not fit them; auto-fit measures changes in idle windows and checks every change against the next real requests |

## What it does

- **Instances.** Run any number of inference servers side by side, each pinned to the GPU(s)
  you pick (one card, another card, both, or CPU-only), each with its own parameters, port,
  logs, statistics and `systemd --user` unit (`LexiPanel-inst@<id>`). The original server stays
  as the legacy `main` instance on its own system unit.
- **Five engines, one workflow.** llama.cpp for text and vision chat, stable-diffusion.cpp
  for images, audio.cpp for TTS, voice cloning, transcription, music and sound effects, and
  Camelid, a Rust GGUF chat engine with its own chat/agent web UI (CPU or NVIDIA), and ONNX
  Runtime for ONNX models on the CPU or an NPU (AMD Ryzen AI, Intel, Qualcomm).
  Create, configure, start, stop and monitor them all the same way.
- **Every parameter, explained.** 220+ settings in grouped forms, each with a plain-English
  tooltip. Anything the form does not cover is still reachable: the *All build options*
  list parses the active build's `--help` and lets you tick any flag.
- **Preview before you launch.** The launch plan shows the exact argv and environment a
  start will use, plus the errors that would refuse it and the warnings that would make it
  slow — before anything runs.
- **Memory you can trust.** A VRAM/RAM estimator that reads the KV-cache shape from the
  model's GGUF header, measures real overhead from the running process, subtracts what other
  servers already hold on the same card, and refuses a start that would run the host out of
  RAM.
- **Fallback tiers.** `normal` → `safe` → `minimal`: after consecutive failed starts, an
  instance automatically falls back to a known-good smaller configuration instead of
  crash-looping.
- **Optimizer.** Benchmarks candidate settings against agentic and coding workloads and
  recommends the winner, with thermal and abort limits so a bad candidate can't cook the
  card. Its models mode compares whole model files the same way. Given a workload mix, speed
  is the time a typical request of that mix takes, measured at the depths it reaches.
- **Workload profile and auto-fit.** Every real request is kept (llama-server wipes its log on
  each restart) and profiled: context depth, output length, concurrency, the idle hours of the
  week, real decode against the measured curve. Findings say where the settings do not fit
  that workload, with the evidence. Auto-fit, off until you turn it on, measures changes in
  idle windows, proposes them or applies speed-only wins by itself, and checks each change
  against the next real requests, rolling back one that made them slower. See
  [Workload and auto-fit](#workload-and-auto-fit).
- **Decode-vs-depth curve.** Measures tokens/s at 8k, 32k, 128k, 240k… context on the
  running model, with GPU temperatures sampled *during* each request — because one "t/s"
  number lies about long-context agent sessions. The live decode chart then judges every
  request against that curve at the request's own context depth.
- **Fit: requants made for your cards.** Plans each tensor's format from the model's
  full-precision source so it fits the VRAM your running setup actually leaves, on the
  formats your cards measured fastest, with an explanation for every choice. Builds on the
  CPU while the server keeps serving, then verifies each candidate against today's model
  (coding and agent suites, decode curve, real VRAM). Also: importance matrices, projector /
  sd.cpp / audio.cpp conversion, and a daily check for new upstream revisions.
- **Refusal check (safety).** Measures how often the running model declines requests it
  should decline and how often it wrongly refuses ordinary ones, using two public prompt
  sets. A strict verdict counts only real first-person refusals; every verdict is listed
  so you can check it. Nothing restarts: it sends one chat request at a time.
- **Builds on tap.** Browse upstream releases of llama.cpp, sd.cpp, audio.cpp and Camelid, install
  any flavour (Vulkan / ROCm / CUDA / CPU) in one click, choose which build each backend
  uses, and optionally let a daily updater fetch new ones (it never restarts a running
  server).
- **Models.** Download from Hugging Face (token support, resumable, verified), see which
  instance uses what, delete safely, extract and lint chat templates.
- **GPU.** Per-card VRAM/GTT, per-process residency from DRM fdinfo, thermals, clocks, PCIe
  link, `amdgpu_top`, and a **power cap** per card that survives reboots.
- **GPU tuning.** Every card in detail, like GPU-Z: ids, VBIOS, DPM clock tables, sensors, the
  OverDrive table. Core and memory clock limits and an undervolt (amdgpu OverDrive), power cap,
  performance level and fan curve, through the power helper. A benchmark on your own model
  measures what each change is worth in tokens/s and tokens per joule; **apply and test** puts
  the old values back by itself when a change turns out unstable. VBIOS backup and a pre-flash
  check of a ROM file. See [GPU Tuning](#gpu-tuning).
- **Power options.** CPU governor and idle states, PCIe and NVMe power saving, GPU performance
  level, fan curve and cap, the hardware watchdog and journal sync, in profiles; a boot profile
  applied before the inference servers start, drift shown, every change logged; boots and how
  they ended, per profile; PSU / UPS budget against the measured GPU peaks.
- **MCP server.** The panel as Model Context Protocol tools, so an AI client (Claude Code,
  Claude Desktop, any MCP client) can check status, read logs, compare settings, benchmark and
  start or stop instances through the same guard rails as the UI. See [MCP server](#mcp-server).
- **Access.** Single-user (one login, as a home box needs) or multi-user, chosen at setup: users
  with roles (viewer, operator, admin), API keys, and a tamper-evident audit log of every change.
  See [Access](#access-single-user-or-multi-user).
- **One gateway.** An OpenAI-compatible `/v1` endpoint for every instance, with per-user quotas
  and usage. See [Gateway and quotas](#gateway-and-quotas).
- **Fleet.** Full LexiPanel on many boxes, one primary that sees them all and serves their shared
  models through its gateway, with replicas and failover. See [Fleet](#fleet-many-boxes-one-primary).
- **Crash forensics.** Boot history with clean-vs-hard-stop triage, the parameters that were
  active at each crash, and a one-click debug bundle written for an AI assistant to read.
- **Files.** A file manager for your home folder, with quick-jump buttons for uploads,
  models & templates, image and audio models and outputs, logs, thermal logs, support
  bundles, backups and launch scripts. Upload files or whole folders (buttons or drag & drop
  from the desktop), download files or zips of folders and selections, new folder, rename,
  delete, copy / cut / paste, drag rows onto folders to move, sortable columns, keyboard
  shortcuts. Uploads stream to disk, so multi-GB GGUFs are fine. Guard rails: hidden system
  folders (`.ssh`, `.config`, …) are unreachable, `panel/` is read-only, and a model,
  projector, template or log that a server is using or a setting points at cannot be
  deleted, moved or renamed.
- **Terminal.** An in-browser terminal (ttyd) behind the same login and your Linux password.

## Tour of the tabs

| Tab | What you get |
|---|---|
| **Status** | Every running server on the box (any engine), start/stop/restart, the instance switcher, throughput hero + live decode chart, decode-vs-depth curve, instance profiles. For image instances a **Generate** card and gallery; for audio instances a **Run** card with per-model options, audio players and recent outputs. For llama.cpp instances a **Hermes Agent** card: readiness checks and its `config.yaml`. |
| **Parameters** | The grouped settings form with tooltips, backend picker, memory calculator, RAM budget, fallback tiers, model profiles, *All build options*. Image instances get model presets; audio instances get **Models on this instance** and the **Model catalog**. ONNX Runtime instances get the provider (CPU or NPU), its options and every generation option of the runtime. |
| **Optimize** | Benchmark suites (agentic, coding), candidate sweeps, thermal/abort limits, recommended settings you can apply. A models mode compares whole model files the same way. |
| **Workload** | The instance's real requests: per day, depth against what a conversation can hold, output length, decode percentiles, concurrency, draft acceptance, real vs measured curve; depth histogram, busy/idle heatmap of the week, findings; **Auto-fit** settings, proposals and the history of every experiment with its check on real traffic. |
| **Fleet** | This box's role (standalone, primary, member). On the primary: every box online / stale / offline with its hardware, instances, speeds and proposals, join codes, revoke. On a member: the primary's address, join, *share this box's models*. |
| **Access** | Single- or multi-user mode, users and roles, API keys (shown once, with expiry), and the audit log with its chain check. |
| **Fit** | Hardware-fitted requants: readiness checklist, per-tensor planner that explains every choice, builds, a report card per candidate (verified on the suites, the depth curve and the VRAM it really uses), format speeds per card, importance matrix, projector / sd.cpp / audio.cpp conversion, upstream watch, **refusal check** (safety evaluation of the running model). |
| **Models** | Everything on disk, size, which instance references it, safe delete. |
| **Download** | Hugging Face downloads with progress; HF token storage. |
| **Templates** | Chat templates on disk, extract from a GGUF, lint for tool-calling / reasoning features, pin to an instance. |
| **GPU** | Devices, memory, per-process residency, thermals & power, clocks & link, VRAM-vs-GTT history, `amdgpu_top`, **Power cap**. |
| **GPU Tuning** | Card details, clock / voltage / power / fan settings with the kernel's accepted ranges and boot defaults, benchmark with live power chart, results compared against a reference run, advice, VBIOS backup and ROM check. |
| **Statistics** | Requests, tokens, decode/prefill medians and history, draft acceptance. |
| **Diagnostics** | Pass/warn/fail checks with evidence and the fix, live command line, debug bundle, configuration backup. |
| **Power options** | Equipment, every power setting (live, boot default, boot profile), profiles, drift, audit log, stability by profile, power budget & UPS. |
| **Crashes** | Boot history, hard-stop triage, parameters in force at each crash. |
| **Builds** | Engine catalogs, upstream releases, install/activate/delete, daily updater policy. |
| **Logs** | Engine and launch logs, auto-following. |
| **Files** | Home-folder manager with quick jumps (Uploads, Models & templates, Logs, Backups…): drag & drop upload of files and folders, download (folders and selections as zip), new folder, rename, delete, copy / cut / paste, drag-to-move, sort, filter, Ctrl+A/C/X/V, Delete, F2. In-use and referenced files are protected. |
| **Terminal** | ttyd in the page (runs Claude Code in the original setup; a plain shell unit is included too). |

## Engines

| Engine | Server | What for | Backends | Instance settings |
|---|---|---|---|---|
| **llama.cpp** | `llama-server` | Chat, coding agents, vision (mmproj), speculative decoding (draft model, embedded MTP, n-gram) | Vulkan, ROCm, CUDA, CPU; multi-GPU via Vulkan | 164 — context & KV cache, offload, RoPE/YaRN, throughput, speculative, sampling, reasoning, vision, Vulkan/ROCm env knobs, server |
| **stable-diffusion.cpp** | `sd-server` | Text-to-image (Qwen-Image 2.1 preset included) | Vulkan, ROCm, CPU | 31 — model files, placement (diffusion / text encoder / VAE per device), offload, VAE tiling, generation defaults |
| **audio.cpp** | `audiocpp_server` | TTS, voice cloning & design, speech-to-text, music & song generation, sound effects, stem separation, VAD, diarization — 80+ model families | Vulkan, CPU | 13 server settings **plus** per-model load / session / default-request options generated from the build's model specs |
| **Camelid** | `camelid serve` | Chat with a curated, validated model list (Qwen3, Llama 3.x, Gemma 4, Mistral, DeepSeek R1 distills, BitNet…), its own web UI, OpenAI-compatible API | CPU, CUDA (NVIDIA). No Vulkan/ROCm, so AMD cards are refused. | 18 — model, threads, KV cache precision, thinking, speculative decoding (n-gram / draft), limits, API key, LAN mode |
| **ONNX Runtime** | `onnx_server.py` (onnxruntime-genai) | Chat with ONNX models (the GenAI format: `genai_config.json` + `model.onnx`), OpenAI-compatible API with streaming. See [ONNX Runtime](#onnx-runtime-npus-amd-intel-qualcomm) | CPU; NPUs: AMD Ryzen AI (VitisAI), Intel (OpenVINO: NPU / GPU / CPU), Qualcomm (QNN); CUDA, DirectML, WebGPU | 32 — model, provider and its options, all 17 generation options the runtime reports (checked at every start), threads, server, API key |

### audio.cpp at a glance

- Pick models from a catalog of every family the installed build supports (Kokoro, Qwen3-TTS,
  VoxCPM2, VibeVoice, Fish Audio, IndexTTS-2, Moonshine, Qwen3-ASR, Parakeet, ACE-Step,
  Stable Audio 3, YuE2, HTDemucs, …), check download sizes, install with one click.
- Each model's options appear as a form with upstream's own descriptions as tooltips —
  weight precision, memory arenas, chunking, seeds, sampling, and so on.
- The **Run** card adapts to the task: text + voice for TTS, a reference WAV (+ transcript)
  for cloning, prompt + lyrics + length for music, an input WAV for transcription and
  separation. Outputs are kept per instance with players and download links.
- Residency controls so audio can share a card with an LLM: models kept loaded, unload after
  idle, and a keep-free memory guard.

Example: Kokoro 82M on an 8-core CPU speaks at ~2.7× real time; Moonshine Tiny transcribes a
6 s clip in ~60 ms.

## Architecture

```
 browser ──HTTPS──▶ Caddy :443 (basic auth, internal CA)
                      ├─ /terminal/*  ──▶ ttyd :8091   (LexiPanel-ttyd)
                      ├─ /shell/*     ──▶ ttyd :8092   (LexiPanel-shell, optional)
                      └─ everything   ──▶ panel :8090  (LexiPanel-panel, 127.0.0.1 only)
                                              │  JSON API + static/index.html
                                              │
          ┌───────────────────────────────────┼─────────────────────────────────┐
          ▼                                   ▼                                 ▼
  LexiPanel-llama.service                 LexiPanel-inst@<id>.service            (per instance)
  run_llama_*.sh               instance_launch.py <id>
  sources panel/params.env            reads instances/<id>/params.env
          │                                   │  launch plan → argv + env
          ▼                                   ▼
     llama-server  :8081           llama-server | sd-server | audiocpp_server | camelid | onnx-server  :808x
```

- `panel.py` is the whole backend (HTTP server, launch plans, estimators, crash triage).
  Engine specifics live beside it: `sdcpp.py`, `audiocpp.py`, `engines.py` (build catalogs +
  updater), `gpupower.py`, `optimizer.py` + `optimize_suite.py`, `depthcurve.py`,
  `flagcatalog.py` + `flaghelp.py`, `workload.py` + `autofit.py`.
- `instance_launch.py` executes **exactly** the plan the UI previewed. It keeps a
  failed-start counter (which drives the fallback tiers), a host-RAM-floor watchdog that
  kills the server before the host locks up, and archives each engine log.
- State is plain files: `params.env`, JSON beside it, per-instance directories. Back it all
  up with `make-backup.sh`.

## Install

Tested on Ubuntu 26.04 with Python 3.14 (stdlib only; `python3-jinja2` is optional and adds a
parse check when saving chat templates), Caddy 2.x, ttyd, systemd.

**Guided:** `bash install-interactive.sh` walks through the steps below in order, asking before
each one, with a time for each. Run `bash install-interactive.sh --check` first: it checks the
account and paths and changes nothing. Re-running it later updates the code and keeps every
settings file, instance and result. Its last, optional step installs the ONNX Runtime engine
(`bash install-onnx.sh`, for ONNX models and NPUs; it offers it by default when it sees an NPU).
The steps by hand:

1. **Put the files in place** as `/home/admin/panel` (and the launch scripts in
   `/home/admin/llama`), owned by `admin`.
2. **Install the front door** (Caddy + basic auth + panel and terminal units):
   ```bash
   bash install-panel-deps.sh   # or: sudo apt install caddy ttyd apache2-utils
   bash install.sh              # asks for a panel username and password, writes /etc/caddy/Caddyfile
   ```
   It prints the URL. The certificate comes from Caddy's internal CA; import its root into
   your browser to silence the warning (the script tells you where it is).
3. **Make the main LLM start on boot** (optional; installs `LexiPanel-llama.service` and a sudoers
   rule scoped to start/stop/restart that one unit):
   ```bash
   bash install-autostart.sh
   ```
4. **Let instances survive logout and start at boot:**
   ```bash
   sudo loginctl enable-linger admin
   ```
5. **Get an engine build:** Builds tab → *Show upstream releases* → Install → *Use for vulkan*
   (or rocm / cuda / cpu).
6. **Create an instance:** Status tab → *New instance* → pick engine, device(s), backend →
   Create → set it up on the Parameters tab → Start.

Optional extras:

| Want | Run once |
|---|---|
| GPU power cap from the panel | `echo 'ACTION=="add|change", SUBSYSTEM=="hwmon", ATTR{name}=="amdgpu", RUN+="/bin/sh -c '\''chgrp admin /sys%p/power1_cap && chmod g+w /sys%p/power1_cap'\''"' \| sudo tee /etc/udev/rules.d/99-LexiPanel-gpu-powercap.rules && sudo udevadm control --reload && sudo udevadm trigger --action=change --subsystem-match=hwmon` (the GPU tab shows this too) |
| Changing settings in Power options and GPU Tuning | `sudo bash power/install-power.sh` (one root helper, one sudoers rule, a boot unit; changes nothing by itself) |
| AMD fan curve, clock limits and undervolt (OverDrive) | copy `systemd/99-amdgpu-overdrive.cfg` to `/etc/default/grub.d/`, `sudo update-grub`, reboot |
| LAN-only raw ports | `bash lockdown.sh` (ufw; edit the subnet first) |
| eSpeak-ng for Kokoro/Piper/Kitten TTS | `sudo apt install libespeak-ng1 espeak-ng-data` |
| ROCm runtime | `sudo apt install libamdhip64-7 librocblas5 libhipblas3` |

## Configuration files

| File | Written by | Read by | What |
|---|---|---|---|
| `params.env` | panel | `run_llama_*.sh` | Settings for the legacy `main` instance. **Overrides the launch script's defaults**, so the script and the running process legitimately disagree. |
| `params-<backend>.env` | panel | panel | Per-backend copies; switching backend swaps them into `params.env`. |
| `params-tier-{normal,safe,minimal}.env` | panel | launch script | Fallback tiers, chosen by the failed-start counter. |
| `instances/<id>/instance.json` | panel | panel, launcher | Name, engine, device(s). |
| `instances/<id>/params.env` | panel | `instance_launch.py` | That instance's settings. |
| `instances/<id>/audio-models.json` | panel | launcher | audio.cpp: models served, task, per-model options. |
| `builds.env` / `engine-builds.json` | panel | launch scripts / launcher | Which build each backend runs. |
| `engine-updates.json` | panel | updater | Daily updater policy per engine (off by default). |
| `gpu-power.json` | panel | panel | Saved power caps, re-applied at start and every 60 s. |
| home folder | Files tab | you | The file manager's root. Set `LEXIPANEL_FILES_ROOT` in the panel's environment to use another one. |
| `profiles/`, `curves/`, `optimize/` | panel | panel | Model profiles, depth curves, optimizer runs. |
| `power-state/` | panel | panel | Power options: saved profiles, settings, the audit log of every change (GPU Tuning changes included). |
| `gpu-tune/`, `gpu-bios/` | panel | panel | GPU Tuning benchmark results; VBIOS backups (`.rom` + what was read from it). |
| `workload/<id>/` | panel | panel | Workload profile: every completed request (`requests.jsonl`, 60 days), hourly slot activity (`activity.jsonl`), configurations seen, auto-fit settings and experiments (`autofit.json`). |

Examples of `params.env` and a tier file are in [`examples/`](examples/).

## Parameters

The complete reference — every key, its default, flag, allowed values and the full tooltip
text — is **[PARAMETERS.md](PARAMETERS.md)** (generated from the code, so it can't drift).

Highlights of the llama.cpp groups:

| Group | Settings | Examples |
|---|---|---|
| Backend | 1 | `BACKEND` vulkan / rocm / cuda / cpu |
| Offload | 9 | `OVERRIDE_TENSORS`, `N_CPU_MOE`, `SPLIT_MODE`, `TENSOR_SPLIT`, `MMPROJ_DEVICE` |
| Context & memory | 18 | `CTX`, `KV_TYPE`, `CACHE_RAM`, `CACHE_REUSE`, `CTX_CHECKPOINTS` |
| Long context (RoPE) | 9 | `ROPE_SCALING`, `YARN_ORIG_CTX`, `YARN_EXT_FACTOR` |
| Throughput | 11 | `NGL`, `BATCH`, `UBATCH`, `THREADS`, `FLASH_ATTN`, `PARALLEL` |
| Speculative decoding | 9 | `SPEC_TYPE` (`draft-mtp`, draft model, n-gram), `SPEC_N_MAX`, `SPEC_P_MIN` |
| Speculative (n-gram) | 12 | n-gram mod / simple / map-k tuning |
| ROCm (HIP) | 9 | `HSA_ENABLE_SDMA`, `GGML_CUDA_*` graph / pinned-memory switches |
| Vulkan (RADV) | 32 | `GGML_VK_DISABLE_COOPMAT`, `GGML_VK_FORCE_MAX_ALLOCATION_SIZE`, … |
| Vision | 6 | `USE_MMPROJ`, `MMPROJ`, `IMAGE_MIN_TOKENS`, `IMAGE_MAX_TOKENS` |
| Generation | 1 | `N_PREDICT` |
| Reasoning | 8 | `REASONING`, `REASONING_EFFORT`, `REASONING_BUDGET`, `PRESERVE_THINKING` |
| Sampling | 25 | `TEMP`, `TOP_P`, `TOP_K`, `MIN_P`, DRY, XTC, mirostat, dynatemp |
| Safety | 1 | `RAM_FLOOR_MB` — the launcher's host-RAM kill switch |
| Server | 13 | `PORT`, `HOST`, `API_KEY`, `ALIAS`, `TIMEOUT`, `WEBUI` |

audio.cpp server settings: `BACKEND`, `AC_THREADS`, `AC_MAX_LOADED`, `AC_IDLE_UNLOAD_S`,
`AC_MIN_FREE_MB`, `AC_LAZY`, `RAM_FLOOR_MB`, `AC_BUSY_TIMEOUT_S`, `AC_UI`, `AC_LOG`, `PORT`,
`HOST`, `AC_EXTRA` — plus each model's own options.

## HTTP API

The UI is a thin layer over a JSON API on `127.0.0.1:8090`; everything the UI does, a script
can do. Add `?inst=<id>` to target an instance. See **[API.md](API.md)** for all 173 routes
and curl examples. The same API is offered to AI clients as tools: [MCP server](#mcp-server).

```bash
curl -s http://127.0.0.1:8090/api/servers | jq                       # what is running
curl -s 'http://127.0.0.1:8090/api/launch-plan?inst=coder' | jq .argv # what a start would run
```

## Safety rails (learned the hard way)

These are enforced in code; each one exists because the box went down without it.

- **No full-size draft model.** `--spec-draft-model` pointed at the target model loads a
  second full copy and hard-locked the host twice. Embedded MTP uses
  `--spec-type draft-mtp` with no draft model; the panel refuses the full-size case.
- **`GGML_VK_ALLOW_SYSMEM_FALLBACK=0` always.** Without it a Vulkan allocation that doesn't
  fit silently lands in host RAM and the model runs at ~1/20th speed with no error. Every
  launch plan sets it; Diagnostics checks it on the live process.
- **Never the iGPU.** Vulkan instances are pinned by ICD; the Intel iGPU is refused as a
  device. audio.cpp CPU instances hide every Vulkan ICD.
- **Host RAM floor.** The launcher kills a server before `MemAvailable` drops below the
  floor, and the RAM budget refuses starts that can't fit alongside what's already running.
- **Fallback tiers** stop crash loops: one failed start → `safe`, two or more → `minimal`; a start that stays up resets the counter.
- **Exit 78 = refused plan.** systemd doesn't retry a configuration error.
- **Start guard for `main`.** Its unit allows 3 starts per 10 minutes (a crash-loop guard), and a
  restart while the model is still loading kills it and counts as a failed start. The panel shows
  starts used, loading state and the next fallback tier on the Server card; a refused start or
  Save-and-restart opens a popup with the reason, when the lockout clears and how to clear it now
  (a **Clear lockout** button once the sudoers rule includes `reset-failed`). Refusals are logged as
  `[start-guard]` in the panel journal and summarised in Diagnostics → `start_guard`.
- **Caddy has no admin API** here (`admin off`): `systemctl reload caddy` always fails, use
  `restart`.
- **The power cap is an average (PPT) limit.** It lowers sustained draw and heat; it does
  not clip sub-second spikes, so it is not a fix for PSU/UPS trips.
- **GPU tuning is live only, and tested.** Clock, voltage and fan changes made on the GPU
  Tuning tab vanish at the next reboot unless you put them in a boot profile yourself, so an
  unstable setting is undone by restarting. Apply-and-test puts the previous values back when
  the run fails, the server dies, the kernel logs a GPU reset or a thermal limit trips. The
  voltage offset is undervolt only. Benchmarks never overlap an optimizer or depth-curve run.
- **No firmware writes.** The VBIOS backup only reads; the ROM check compares a file with the
  card and ends with the vendor tool's command for you to run. (`flashrom -p internal`, which an
  early attempt at this feature used, programs the *motherboard's* flash chip, not the GPU's.)

## Security

- **Cross-site protection.** Browsers resend basic-auth credentials to requests started by
  *any* website, so the API refuses requests a browser marks as cross-site, a foreign
  `Origin`, and form / `text/plain` POSTs (see [API.md](API.md)). The terminal units run
  ttyd with `-O` so a foreign page cannot open the terminal websocket either.
- **Settings files are written single-quoted.** `params.env`, the tier files and
  `builds.env` are sourced by bash; values are stored literally, so `$(...)`, backticks and
  quotes in a setting are data, never code. Newlines are refused.
- The panel binds **127.0.0.1** only. Caddy in front provides TLS and basic auth. Both
  Caddyfiles here ship with `$2a$14$REPLACE_ME`; `install.sh` fills in your hash.
- **llama-server, sd-server and audiocpp_server have no authentication.** Keep instances on
  `127.0.0.1` unless you mean it; a `0.0.0.0` bind is flagged as a warning. `lockdown.sh`
  restricts the raw ports to your LAN with ufw. llama.cpp's `API_KEY` setting is available.
- **The terminal asks you to sign in.** ttyd starts `su -l admin`, so every terminal session
  (including a page refresh) asks for the Linux account password, checked by PAM, before
  any shell exists. That is on top of the panel login. Traffic is TLS end to end from the
  browser to Caddy (HTTP redirects to HTTPS); Caddy to ttyd is loopback only. Import Caddy's
  root certificate in your browser so you can tell the real box from an impostor. Remove the
  `/terminal` route from the Caddyfile if you don't want a web terminal at all.
- The sudoers rules are scoped: start/stop/restart of `LexiPanel-llama`, and the power
  helper's five verbs (`status`, `apply`, `persist`, `unpersist`, `vbios`). The helper checks
  every target against devices it found itself and every value against that device's own range.
- **MCP.** `/api/mcp` sits behind the same Caddy login and cross-site protection as the rest of
  the API. Every tool is a documented route, so an AI client can do only what the UI can; there
  is no file-write tool, and `LEXIPANEL_MCP_READONLY=1` hides every tool that changes anything.

## Adapting it to your box

This is one machine's panel, generalised only as far as renaming the user:

- **User and paths:** `admin`, `/home/admin/{panel,llama,models,sdcpp,audiocpp}`. To use
  another user: change `HOME` at the top of `panel.py`, the `User=`/paths in `systemd/*`,
  the sudoers file, and the launch scripts (`grep -rn /home/admin`).
- **Legacy `main` instance:** launched by `launch-scripts/run_llama_vulkan.sh`
  (also `_rocm`, `_cpu`), written for one Qwen 27B model with embedded MTP. Point `MODEL`
  at yours in `params.env`, or ignore `main` and use instances only.
- **Hardware assumptions:** AMD first (amdgpu sysfs for VRAM, thermals, power, OverDrive);
  NVIDIA supported via `nvidia-smi` and Mesa NVK; tooltips quote measurements from a
  7900 XTX. Multi-GPU is Vulkan-only.
- **Ports:** panel 8090, terminals 8091/8092, main LLM 8081, telemetry 8082, instances from
  8083 up.

## Workload and auto-fit

The Optimize tab answers "which settings win on the benchmark?". The **Workload** tab answers
"what do this instance's real requests look like, and do the settings fit them?", and
**auto-fit** keeps asking as the workload changes. llama.cpp instances only.

**The profile** (`workload.py`, always on, nothing new is polled):

- Every completed request is read from the engine log every 30 s into
  `workload/<id>/requests.jsonl`: new prompt tokens, output tokens, decode and prefill rate,
  draft acceptance, the context depth it ran at, and a fingerprint of the configuration that
  served it (the running argv, less port and log file). llama-server truncates its log on every
  restart; the store keeps 60 days. On first start the archived engine logs are read once for a
  head start.
- The status sampler's `/slots` polls (every 2 s, already running) are summed per hour: how much
  of the hour a request was in flight, and how many slots were busy at once.
- LexiPanel's own measuring (optimizer, depth curve, GPU benchmark, refusal check, Fit) stays out
  of both: requests completed while it runs are tagged and left out, and its own in-flight
  requests do not count as busy. An experiment at 3 am must not teach it that 3 am is busy.
- On the tab: requests per day; depth p50 / p90 / deepest against what one conversation can
  hold; output length; decode percentiles; the most slots busy at once; draft acceptance; real
  decode against the measured depth curve. A depth histogram with each band's share of
  generated tokens (where decoding time goes). A heatmap of the 168 hours of the week, local
  time, with the learned idle windows marked; a table view has every value.

**Findings** need 50 requests over 3 days, and each quotes its evidence:

| Finding | When | What auto-fit can measure |
|---|---|---|
| Context reserved but never reached | the deepest request is at most half of a conversation's context | `CTX` = deepest + 25 %, rounded up to 16k, with the KV cache it frees |
| Sessions reach the context limit | 3 or more requests at 95 % of it | `CTX` + 16k per slot, if the estimator says it fits |
| Parallel slots never used | `PARALLEL` > 1, never more than one busy in 7 days of watching | `PARALLEL=1` |
| Every slot busy at peak times | several slots, all busy 30+ minutes a day | reported, not tried |
| Speculative decoding may not pay | median draft acceptance under 0.45 over 50+ requests | `SPEC_TYPE=none` |
| Real requests run below the measured curve | the last 3 days under 85 % of the curve at the same depths | the usual causes: throttling, another process on the card, a new build |
| The last configuration change made requests slower / paid off | 20+ requests matched by depth, under 93 % / at least 103 % of before | |

**Auto-fit** (`autofit.py`, **off** by default, per instance). An experiment is an ordinary
optimizer run given the real workload: its depth mix (three bands, weighted by generated
tokens), its typical new prompt and output length, and its p90 depth. Every candidate's decode
is measured at those depths, and speed is the time a typical request of yours takes: its prompt
at the measured prefill rate plus its output at the decode rate over the mix. A knob that is
faster at depth 0 but slower where your sessions run does not win.

- **tune**: the launch speed knobs (`UBATCH`, `BATCH`, speculative draft length and threshold,
  cache reuse, threads), the quick budget's neighbours of the current values. Due when the running
  configuration was never tuned for this workload, after 30 days, or when the p90 depth moves by
  half (8k at least).
- **reshape**: the findings' candidates (context size, slots, speculation on or off), each
  measured as an alternative to the current settings. One whose point is capacity (context
  freed or gained, one conversation given the whole context) is proposed when it is no slower
  (97 % of the current speed or better) at the same quality; turning speculation off only when
  it is the margin faster.

An experiment starts only when auto-fit is on, the instance is running, nothing else is
measuring or restarting, no request came in for 15 minutes (`quiet_min`), now is inside an idle
window with an hour of it left, no earlier change is still being verified, and fewer than 2
(`max_per_week`) ran in the last 7 days. The idle window is learned (an hour of the week
watched at least 1.5 hours over two weeks and under 2 % busy) or set (`02-06`, local time). A
real request arriving stops the experiment: the optimizer lets that request finish on the trial
configuration, then restarts on the saved settings, restored byte for byte. So does running past
120 minutes (`max_minutes`). **Tune now** and **Measure the findings now** run one by hand,
skipping only the window, quiet and weekly checks.

What happens to a winner:

- **propose** mode: it becomes a proposal with **Apply**, **Apply and restart** and **Dismiss**.
- **auto** mode: applied by itself only when it changes speed knobs only (`UBATCH`, `BATCH`,
  `CACHE_REUSE`, `SPEC_N_MAX`, `SPEC_P_MIN`, `THREADS`), a typical request is at least 3 %
  (`min_gain`) faster, quality on the task suite is no lower, and it was never rolled back on
  this configuration. Anything that changes what the server can do (context, slots,
  speculation on or off) is always a proposal. The restart onto the new settings waits for two
  quiet minutes.
- **Verified on real traffic**, whoever applied it: once the server runs the new configuration,
  its real requests are compared with the 14 days before at the same depths (the median decode of
  each depth band, weighted by where the new requests land). At 20 matched requests, under 93 %
  of before is *regressed*: auto mode rolls it back and restarts, and never auto-applies that
  change on that configuration again; propose mode reports it with a **Roll back** button.
  Otherwise *confirmed*. Too little traffic in 7 days is *inconclusive*. A rollback leaves alone
  any setting you changed by hand since.

It never swaps the model, never touches GPU clocks, voltage or power, runs one experiment at a
time, and never runs through real traffic. The limits worth knowing: quality is LexiPanel's
task suite, not your prompts; learned idle windows assume a weekly rhythm; the depth of a
request is the context it released at (history + new prompt + output). Over MCP:
`get_workload` and `apply_workload_proposal`.

To watch the loop work without touching anything, `bash tests/e2e/run.sh confirm` (or
`regress`, `interrupt`) runs it on a temporary copy of the panel's code against a stand-in
llama-server, in about three minutes; `HOLD=120` keeps the copy's UI up at
`http://127.0.0.1:18290/` at the end. It uses no GPU and none of your settings.

## GPU Tuning

One tab for the question "what is this card worth on my model, and what does a change buy?".

- **Card**: PCI ids, board, VBIOS version, VRAM size and vendor, PCIe link now and max, DPM clock
  tables with the current level, edge / junction / memory temperatures against their limits,
  fan, board power against its cap, core voltage, performance level, power profile, and the
  OverDrive table with the range the kernel accepts. Plain sysfs; nvidia-smi for NVIDIA.
- **Settings**: max core clock, max memory clock and core voltage offset (amdgpu OverDrive,
  RDNA2 and RDNA3), power cap, performance level, fan curve. Each row shows the live value, this
  boot's default and the accepted range. They are the power helper's knobs, so the helper
  validates, writes, reads back and logs them (Power options audit), and they can go in a Power
  options boot profile once proven. Needs the helper; clocks and voltage need OverDrive
  (Install, optional extras).
- **Benchmark**: one fixed llama.cpp request (default 2048 prompt tokens + 256 generated, 5 runs)
  on a running instance of the card, with power, clocks, temperatures and load sampled every
  second. Decode t/s and its spread between runs, cold prefill, mean and peak power, tokens per
  joule, peak junction and memory, and GPU resets from the kernel log. Every result keeps the
  settings it ran with and is compared against a reference run (the oldest stable one unless
  you pick one).
- **Apply and test**: apply the settings, run the benchmark, and put the previous values back
  if anything fails. The change stays only if the run was stable.
- **Profiles**: save the current fan / voltage / clock set as a named profile, test it with the
  benchmark, and set the winner to apply at every boot (stored with the Power options profiles,
  applied before the inference servers start). This is how a proven overclock or undervolt is
  made to survive a reboot; a change made without saving it to a boot profile is gone at restart.
- **Advice**: power-limited, thermally limited, not GPU-bound, noisy runs, fastest and most
  efficient run so far; each line quotes the number it is based on.
- **VBIOS**: back up the card's image (read through the helper; amdgpu's debugfs copy or the PCI
  ROM), and check a ROM file before flashing it: signature, vendor and device id against the
  card, size, UEFI image, board part number, and whether it differs from the backup. LexiPanel
  prints the vendor tool's command (`amdvbflash`, `nvflash`) and never runs it.

LLM decode reads every weight for each token, so it is bound by memory bandwidth
([fit/FINDINGS.md](fit/FINDINGS.md)): memory clock tends to move tokens/s, core clock mostly
moves watts. An undervolt usually buys efficiency. The benchmark is there so you don't have to
take either on faith.

## MCP server

`mcp_server.py` offers the panel to AI clients as [Model Context Protocol](https://modelcontextprotocol.io)
tools: 28 of them (instances, status, parameters with a memory estimate before saving, launch
plans, models, GPU and GPU Tuning, benchmarks, depth curves, optimizer status, the workload
profile and auto-fit proposals, statistics, diagnostics, crashes, logs, power options, and
reading text files from the home folder). Each
tool is a call to a documented route, so the panel's own checks apply; a refused start comes
back with the reason. Stdlib only.

```bash
# stdio, for clients that start a command; run on the box, or through ssh
claude mcp add lexipanel -- ssh admin@box python3 /home/admin/panel/mcp_server.py

# Streamable HTTP, through Caddy with the panel login
claude mcp add --transport http lexipanel https://box/api/mcp \
    --header "Authorization: Basic $(printf 'admin:PASSWORD' | base64)"
```

`LEXIPANEL_MCP_READONLY=1` (in the panel's environment, or the stdio server's) leaves only the
tools that read. For a script, `GET /api/mcp/tools` and `POST /api/mcp/call {"name","arguments"}`
are the same tools as plain JSON. `python3 mcp_server.py --help` lists the environment
variables for reaching the panel through Caddy from another machine.

### Hermes Agent

[Hermes Agent](https://hermes-agent.nousresearch.com/) (Nous Research) can use a llama.cpp
instance as its model and these MCP tools as its hands. The **Hermes Agent** card on the Status
tab checks what Hermes needs of the instance: started with `--jinja` (LexiPanel always passes it;
without it llama-server ignores Hermes' tools), at least 64,000 tokens per conversation (`CTX` /
`PARALLEL`), a chat template that handles tools, and an address Hermes can reach (`HOST=0.0.0.0`
if it runs on another machine). It then writes the `model:` and `mcp_servers:` blocks for
`~/.hermes/config.yaml`, pointed at this instance, with the MCP tools read-only
(`LEXIPANEL_MCP_READONLY=1`) until you remove that line. An API key, if the instance has one, is
referenced as `${env:LEXIPANEL_LLM_KEY}`, never written out. `GET /api/hermes` returns the same.

## GG: Graph Gauntlet

Every chart card has a **▶ GG** button. It swaps the chart for a small runner game whose first
stretch of track **is that chart's data**: the decode rate, the VRAM line, the depth curve.
- **Drag to draw straight lines:** bridges over gaps, ramps to jump spikes and reach pickups,
  roofs to catch the archers' arrows.
- **Lines are limited.** Each comes back once it's 20 m behind you. **+ LINE** and
  **+ LENGTH** pickups give you more.
- Keys: P pauses, R restarts, Esc goes back to the graph.

- **Beat your own numbers.** The game-over screen shows this run against your personal bests
  (metres, archers passed, arrows blocked and dodged, archers tackled, pickups) and marks each
  **NEW BEST**, with this session's runs and metres. **15 achievements**, from *First steps*
  (100 m) and *Archer avoider* (pass 15 archers in a run) to *Kilometre club*, *Artful dodger*,
  *Shield wall* and *Road warrior* (10 km in all); the screen names the next two to go for.

It costs nothing when you're not playing: the loop stops when paused, closed, on another tab,
or in a hidden browser tab. It makes no network requests and uses no libraries, running in your
browser, not on the GPU box. Your record, bests and achievements are kept in the browser. Sources, the installer
and tests are in [`gg/`](gg/). [`gg/GG-AI-GUIDE.md`](gg/GG-AI-GUIDE.md) explains how to rebuild or
extend it without breaking the panel.

## Fit: hardware-fitted requants

Take a model's full-precision source and choose each tensor's format so it fits **your**
cards at the context and speed you need, without losing agentic or coding quality, using the
formats those cards actually run fastest. The **Fit** tab does it in steps:

1. **Source** (phase A, `fit/phaseA-source.sh`): download a pinned revision, check every
   file's SHA-256, convert to a BF16 GGUF.
2. **Measure formats** (phase B): one test file per format, llama-bench on each card of an
   instance, folded into a per-card speed model (decode time = fixed cost + bytes / bandwidth).
3. **Plan** (phase C): the budget is what the running model's weights use plus the VRAM free
   right now, minus a margin, so it is calibrated against a config known to fit. The base is
   the card's fastest measured format; spare bytes go where published sensitivity says they
   buy the most (output layer, attention values, FFN down, first and last layers, MTP head).
   Three goals: same quality but faster, same speed but better, best that fits. Sizes are
   exact: llama-quantize's dry run agrees within 2 MiB.
4. **Build and verify** (phase D): quantize on the CPU at low priority; verify each
   candidate against today's model with the optimizer's suites, a decode curve by depth and
   the VRAM it really occupies.
5. **Parts** (phase E): output layer, MTP head and embeddings are chosen separately; the
   projector, stable-diffusion.cpp and audio.cpp models convert here too.
6. **Upstream watch** (phase F): a daily check for a newer source revision or quantizer. It
   reports; it never downloads.

Quality is ranked with priors until KL divergence is measured, and every plan says so. See
**[fit/README.md](fit/README.md)** and the worked example **[fit/FINDINGS.md](fit/FINDINGS.md)**
(Qwen3.8-27B on a 7900 XTX: IQ4_XS is the fastest and smallest decoder there; decode is almost
purely bytes read).

## macOS (experimental)

`hostos.py` gathers every question LexiPanel asks the operating system (memory, process
details, listening ports, GPU, run folders, and the service manager: systemd on Linux,
launchd on macOS). On Linux it returns exactly what the panel computed before. **The macOS
side was written from Apple's documentation and has never run on a Mac.** Linux-only
features (AMD power caps, sysfs thermals, DRM residency, journald crash triage, Vulkan
pinning) report "not available on macOS". Reports from Apple Silicon owners are welcome.

## ONNX Runtime: NPUs (AMD, Intel, Qualcomm)

A fifth engine for **ONNX models** (the ONNX Runtime GenAI format: a folder with
`genai_config.json` and `model.onnx`), on the CPU or through ONNX Runtime's execution providers:
**AMD Ryzen AI NPU** (`vitisai`), **Intel NPU / GPU / CPU** (`openvino`, pick the device),
**Qualcomm Hexagon NPU** (`qnn`), plus `cuda`, `dml` (Windows) and `webgpu`. llama.cpp does not run
on these NPUs; this is how LexiPanel does. Create an instance with engine *ONNX Runtime*,
set the model folder and the provider on the Parameters tab, Start. It serves an
OpenAI-compatible API (`/v1/chat/completions`, streaming, `stop`, `seed`).

- **Install the runtime once**: `bash install-onnx.sh` (CPU), or `--cuda` / `--openvino` / `--qnn`.
  AMD's VitisAI provider comes with AMD's Ryzen AI Software; point the instance's *Runtime
  Python* at the Python it installs. Provider packages change between releases: if the
  Parameters tab still lists a provider as missing from the build, follow the vendor's ONNX
  Runtime GenAI instructions and point *Runtime Python* there.
- **Every generation option the runtime has, with a tooltip**: the 17 search options
  onnxruntime-genai reports (`max_length`, `do_sample`, `temperature`, `top_k`, `top_p`,
  `repetition_penalty`, beams, `random_seed`, `chunk_size`, ...), empty meaning the model's own
  `genai_config.json` value. At every start the server sets each one and reads it back; the
  Status tab and `/props` say how many this build accepted and which it did not. Provider
  options (OpenVINO `device_type`, QNN `backend_path` / `htp_performance_mode`, VitisAI
  `config_file`, anything else as JSON), a plugin provider library, CPU threads, port, address,
  model name and API key (kept in a 0600 file, never on the command line).
- **Readiness**: the Parameters tab shows the runtime version, which providers the installed
  build has, and the NPUs the kernel sees (`/sys/class/accel`: `amdxdna`, `intel_vpu`, Qualcomm
  `qaic` / fastrpc). Starting refuses a missing runtime, a folder that is not a GenAI model, a
  provider the build lacks, QNN without its library, a LAN listener without an API key.

Verified here on the CPU provider (Qwen2.5-0.5B-Instruct int4, onnxruntime-genai 0.16: all 17
options accepted and read back, chat and streaming through the panel). The NPU providers
could not be run without the hardware: the first run on a Ryzen AI, Core Ultra or Snapdragon
machine is their real test.

## Access: single-user or multi-user

Chosen at setup (`install-interactive.sh` asks), changeable on the **Access** tab.

- **Single-user** (default): as before, Caddy's one login in front, the panel trusts what reaches it.
- **Multi-user**: the panel checks every request itself, so even a local process needs a login.
  Users (scrypt-hashed passwords) log in with the browser's own prompt; scripts and MCP clients
  use **API keys** (`Authorization: Bearer lp_...`; `LEXIPANEL_API_KEY` for `mcp_server.py`),
  each capped at its user's role, with an expiry. **viewer** reads; **operator** also starts,
  stops, restarts and runs benchmarks, the optimizer, depth curves and auto-fit experiments;
  **admin** everything else (settings, parameters, power, GPU tuning, files, builds, fleet,
  users). Any change not listed for operators is admin-only, and a test fails the build if a
  route is ever open to viewers. MCP tool calls run with the caller's role. Every change is in the
  hash-chained **audit log**. Caddy: `systemd/Caddyfile.multi-user.new` (its login stays only in
  front of the terminals). Locked out: `python3 auth.py mode single` as the panel's account.

## Gateway and quotas

One OpenAI-compatible endpoint on the panel for every running llama.cpp, ONNX Runtime and Camelid
instance: `GET /v1/models`, `POST /v1/chat/completions` (streaming passes through), routed by
`model` (instance id, alias or model file name). Clients use the panel's address and a user's API
key; each instance's own key stays inside. Per-user quotas (admin: `POST /api/gateway/quota`
`{user, rpm, tokens_day, concurrent, models}`), refused with HTTP 429 and the reason; usage per day,
user and model at `GET /api/gateway`. Counts only: prompts and replies are never stored.

**Across the fleet**: on a member, tick *share this box's models* on the Fleet tab. Its running
instances that listen on the LAN without an API key are then reported to the primary, and the
primary's gateway serves them too: the same model on several boxes becomes replicas, each request
goes to the one with the fewest open requests, and a replica that does not answer is skipped for
the next. Offline boxes drop out on their own.

## Fleet: many boxes, one primary

Full LexiPanel on every box; one is the **primary** and sees them all in its **Fleet** tab. On the
primary: role *Primary*, then **Add a box** gives a one-time join code (30 minutes). On each other
box: role *Member*, the primary's address, the code, Save. Members then report every minute,
outbound only: hardware (GPUs, NPUs, RAM), instances with engine, state, model and decode speed,
workload counts, pending auto-fit proposals, LexiPanel version. Never prompts, settings or keys.
The primary lists every box as online, stale or offline, and can **Revoke** one. It cannot change
anything on a member: each box is still run from its own panel. Caddy lets the two fleet routes
(`/api/fleet/report`, `/api/fleet/join`) through without the login; the panel checks the box's
token or the join code itself (see `systemd/Caddyfile.new`). Design and next steps, including
striping one large model across boxes with llama.cpp RPC: [docs/FLEET.md](docs/FLEET.md).

## Checks before an upload

`bash tests/run_all.sh` runs everything GitHub runs (about 6 minutes; `--quick` for compile, unit
tests and script syntax in under one) and ends with **OK to upload** or the failures. The browser
checks need node and, once, `cd tests/ui && npm install && npx playwright install chromium`;
without them that part says SKIP. `tests/test_safety.py` is the tripwire for what LexiPanel must
never do: run a firmware flasher, write `/sys` outside the root helper, widen the sudoers rules.

## File layout

```
panel.py              backend: HTTP API, launch plans, estimators, crash triage
instance_launch.py    runs one instance: plan → server, failure counter, RAM watchdog
engines.py            build catalogs (llama.cpp, sd.cpp, audio.cpp, Camelid) + daily updater
sdcpp.py              stable-diffusion.cpp instances: params, presets, jobs, gallery
camelid.py            Camelid instances: params, catalog pulls, launch plan, chat proxy
hostos.py             OS layer: Linux (tested) and macOS (experimental) answers in one place
audiocpp.py           audio.cpp instances: params, model catalog/installs, runs, outputs
gpupower.py           per-GPU power caps, saved and re-applied
gputune.py            GPU Tuning: card details, tuning settings via the power helper, benchmark, VBIOS
poweropts.py          Power options: equipment, settings, profiles, boot profile, stability, UPS
power/                the root helper (lexipanel_power.py), its installer, sudoers rule and boot unit
mcp_server.py         MCP tools for AI clients: stdio server and the /api/mcp handler
hermes.py             Hermes Agent: readiness checks and its config.yaml for an instance
onnxrt.py             ONNX Runtime engine: parameters, NPU detection, launch plan
onnx_server.py        its OpenAI-compatible server (onnxruntime-genai), options verified at start
install-onnx.sh       the runtime in ~/onnxrt/venv (--cuda, --openvino, --qnn)
webui.py              llama.cpp's built-in web UI per instance
filemgr.py            Files tab: home-folder browse, streamed upload/download, zip, copy/move, guard rails
optimizer.py          benchmark-driven parameter search (+ models mode for Fit, workload-weighted runs)
workload.py           workload profile: request store, hourly activity, envelope, findings
autofit.py            auto-fit: idle-window experiments, keep / propose / apply, verify, roll back
fitquant.py           Fit: size tables, card speed models, solver, build/bench/imatrix/verify jobs
refusals.py           refusal check: declines vs over-refusals of the running model, strict + keyword verdicts
optimize_suite.py     the agentic / coding benchmark prompts
depthcurve.py         decode tokens/s vs context depth, with in-request thermals
flagcatalog.py        every flag of the active build, parsed from --help
flaghelp.py           plain-English help for those flags
main_devices.py       device selection for the legacy main instance
static/index.html     the whole UI (single page, no build step)
gg/                   Graph Gauntlet: game source, apply script, tests, AI rebuild guide
fit/                  Fit: phase A/B scripts, README, worked-example findings
systemd/              units, sudoers rule, Caddyfile template, GRUB OverDrive snippet
launch-scripts/       run_llama_{vulkan,rocm,cpu}.sh, bench_backends.sh
examples/             sample params.env and tier file
tests/                python3 -m unittest discover tests (fake sysfs, no GPU needed);
                      tests/e2e/run.sh: auto-fit end to end against a stand-in llama-server;
                      tests/run_all.sh: every check, run before uploading (tests/ci/, tests/ui/)
.github/workflows/    the same checks on GitHub, on every push
install-interactive.sh  guided install of everything below (--check changes nothing)
install-panel-deps.sh required packages, plus --rocm --tts --jinja --ups
install.sh            Caddy + panel + ttyd units
install-autostart.sh  main LLM on boot + scoped sudoers
lockdown.sh           ufw: raw ports LAN-only
fix-firewall.sh       add allow rules safely, with a dead-man rollback
make-backup.sh        tarball of every hard-to-reproduce config (no model weights)
audit-flags.sh        check the launch script's flags against the build's --help
start.sh              run the panel by hand (dev)
PARAMETERS.md         every setting, generated from the code
API.md                every route, generated from the code
CHANGELOG.md          release notes
auth.py               Access: single / multi-user, users, roles, API keys, audit chain
gateway.py            one /v1 endpoint for every instance, per-user quotas, usage
fleet.py              Fleet: roles, join codes, per-box tokens, reports to the primary
docs/FLEET.md         Fleet design: what v1 does, what comes next (striping a model across boxes)
docs/ROADMAP.md       planned after 1.0.0: users and roles, gateway and quotas, fleet serving, SOC 2 readiness
```

## Credits

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp),
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp),
[audio.cpp](https://github.com/0xShug0/audio.cpp) and [ggml](https://github.com/ggml-org/ggml),
fronted by [Caddy](https://caddyserver.com) and [ttyd](https://github.com/tsl0922/ttyd).
Model weights keep their own licenses.

## License

[MIT](LICENSE) © 2026 W61k3r
