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
- [Where LexiPanel is unusually hard to match](#where-lexipanel-is-unusually-hard-to-match)
- [Competitive reality](#competitive-reality)
- [What it does](#what-it-does)
- [Tour of the tabs](#tour-of-the-tabs)
- [Engines](#engines)
- [Architecture](#architecture)
- [Install](#install)
- [Parameters](#parameters)
- [Workload and auto-fit](#workload-and-auto-fit)
- [Fit: hardware-fitted requants](#fit-hardware-fitted-requants)
- [GPU Tuning](#gpu-tuning)
- [Gateway and quotas](#gateway-and-quotas)
- [Access: single-user or multi-user](#access-single-user-or-multi-user)
- [Fleet: many boxes, one primary](#fleet-many-boxes-one-primary)
- [MCP server](#mcp-server)
- [GG: Graph Gauntlet](#gg-graph-gauntlet)
- [HTTP API](#http-api)
- [Safety rails](#safety-rails-learned-the-hard-way)
- [Security](#security)
- [macOS (experimental)](#macos-experimental)
- [Adapting it to your box](#adapting-it-to-your-box)
- [Configuration files](#configuration-files)
- [Checks before an upload](#checks-before-an-upload)
- [File layout](#file-layout)
- [What LexiPanel is not](#what-lexipanel-is-not)
- [Why keep building this instead of gluing tools together?](#why-keep-building-this-instead-of-gluing-tools-together)
- [Roadmap direction](#roadmap-direction)
- [Credits](#credits)

---

## The idea

**Fit AI to the machine; don't just run AI on it.** Most local-AI tools answer *"how do I run
this model?"*. LexiPanel asks *"on this exact machine, with this model and this workload, which
configuration actually works best, and how do we prove it?"*, and answers with measurements
rather than rules of thumb:

| Step | Where it happens |
|---|---|
| **Discover** | GPU and GPU Tuning tabs (cards, clocks, thermals, power, VBIOS), Power options (equipment), NPU/provider readiness, the GGUF reader behind the memory estimator |
| **Fit** | VRAM/RAM estimator and RAM budget, launch-plan preview, Fit tab requants planned per tensor for your cards |
| **Optimize** | Optimize tab, decode-vs-depth curve, GPU Tuning benchmark |
| **Validate** | coding and agent suites, refusal check, Diagnostics, apply-and-test with automatic revert |
| **Operate** | instances, fallback tiers, start guard, gateway, access control, fleet, crash forensics, power profiles |
| **Learn** | statistics and live decode judged against the measured curve, workload depth/concurrency/idle windows, stability per power profile, benchmark history |
| **Adapt** | Auto-fit measures changes in idle windows and checks every change against the real requests that follow |
| **Explain** | exact launch argv/env, refusal reasons, diagnostics evidence, per-tensor Fit reasoning, audit trail and debug bundles |

The individual pieces are useful. **The loop between them is the point.** The memory estimator
knows what other managed servers already occupy. The optimizer knows where real sessions spend
their tokens. Fit knows the VRAM budget of a configuration that has already proved it can run.
GPU Tuning measures the model actually served. Auto-fit checks benchmark wins against later
production traffic. Crash triage knows what configuration and power state were in force.

## Where LexiPanel is unusually hard to match

**Competitive check: September 2026.** The projects below are strong and several are better than
LexiPanel at their own layer. What I have not found in their public code/documentation is another
self-hosted local-AI control plane that combines **all** of the following in one measured system.
If there is one, open an issue — it deserves a link here.

| Capability | What LexiPanel actually does | Closest overlap I found |
|---|---|---|
| **Closed-loop optimization from real traffic** | Keeps real request depth/length/concurrency history, learns idle windows, benchmarks candidates against that workload shape, can auto-apply narrowly safe speed wins, then verifies them against later real requests and rolls back regressions | [Llama Optimizer](https://github.com/VykosX/Llama-Optimizer) has a deeper dedicated Bayesian parameter search; [LumaBrowser](https://www.lumabyte.com/advanced) has strong hardware-aware fitting. Neither public design describes this same long-running traffic → experiment → later-traffic verification loop |
| **Hardware-fitted tensor-level requantization** | Measures quant formats on the actual cards, derives a real VRAM budget from a known-running setup, plans formats per tensor/role, builds from full-precision source and verifies the result against today's model | Quantizers and runtime planners choose whole-model formats well; I did not find this end-to-end per-tensor, per-machine compile-and-verify loop integrated into another local-AI operator plane |
| **Inference-aware electrical tuning** | Changes supported GPU clocks/undervolt/power/fan settings, benchmarks the running LLM, measures tokens/s **and tokens/joule**, watches resets/thermals and automatically restores failed trial settings | GPU tuning suites exist and inference benchmarks exist; I did not find another audited local-AI panel coupling them this tightly |
| **Long-context performance as a first-class operating signal** | Measures decode across context depths and judges live requests against the curve at the depth each request actually reached | Many tools benchmark throughput or size context to fit; few make depth-dependent performance part of continuous operations and later tuning decisions |
| **AI-ops over MCP without a second privilege model** | 28 MCP tools expose launch plans, memory estimates, workload/Auto-fit, GPU tuning, diagnostics, crashes, power and lifecycle through the same API checks and roles as the browser | MCP is common for giving models application tools; this is MCP used as the guarded operator interface to the machine running the models |
| **One evidence chain from model file to physical box** | Model metadata → fit → launch argv/env → process residency → workload → optimizer → GPU/power state → crash evidence → rollback history | Other projects cover parts of this chain extremely well; the breadth of one shared evidence model is the unusual part |

That is the claim LexiPanel should make loudly: **not that every individual subsystem is the best
subsystem in existence, but that the integrated machine-fitting loop is unusually complete.**

There are also several combinations that are rare enough to be meaningful on their own:

- **Real-workload Auto-fit + real-traffic verification + rollback.** Benchmarking is easy to fake by
  choosing the wrong workload. LexiPanel makes the later real requests the final judge.
- **Per-tensor Fit + card-measured quant speeds + live VRAM budget.** The output is not merely a
  recommendation for `Q4_K_M`; it is a build recipe tied to the cards it is meant to run on.
- **LLM benchmark + GPU electrical controls + tokens/joule + automatic revert.** Hardware tuning is
  evaluated in the unit that matters to the workload.
- **Crash loops, RAM floors, silent Vulkan system-memory fallback, power profiles and firmware
  boundaries treated as AI-serving problems rather than somebody else's problem.**

## Competitive reality

Bragging is useful only if the boundaries stay visible. These projects remain stronger in specific
areas, and LexiPanel is better positioned **with** many of them than against them:

| Project | Where it is stronger | Where LexiPanel is stronger/different |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | The inference engine itself; fastest-moving GGUF/runtime feature surface | Operates llama.cpp as part of a measured machine: fit, lifecycle, workload history, hardware/power and safety |
| [Open WebUI](https://github.com/open-webui/open-webui) | Chat UX, RAG/knowledge, collaboration, SSO/OIDC/LDAP/SCIM and application-layer extensibility | Deeper below the API boundary; Open WebUI is a natural frontend for LexiPanel rather than something LexiPanel should reimplement |
| [LumaBrowser](https://www.lumabyte.com/) | Cross-platform packaging, guided hardware-aware setup, browser automation, agent UX, multiple runtimes and polished GPU placement/hot-swap behavior | Deeper Linux host control, real-workload feedback, tensor-level Fit, GPU electrical tuning, power/crash evidence and guarded server operations |
| [Llama Optimizer](https://github.com/VykosX/Llama-Optimizer) | GP/Bayesian search, topology/context sweeps and dedicated MTP/IK-llama optimization methodology | Broader closed loop around optimization: ongoing workload observation, safe apply/propose policy, later real-traffic verification and rollback |
| [llama-swap](https://github.com/mostlygeek/llama-swap) | Generic, mature demand-driven hot swapping of arbitrary OpenAI/Anthropic-compatible servers | Much deeper machine fit, measurement, model build/quant, host safety and hardware operation |
| [LocalAI](https://localai.io/) | 60+ backends, modality breadth, backend plugins, federation, production distributed mode and model sharding | Fewer engines but much deeper bare-metal tuning and evidence on one machine |
| [Lemonade](https://github.com/lemonade-sdk/lemonade) | AI-PC/NPU experience, AMD-optimized heterogeneous execution, embeddable local server and broad desktop-platform support | Deeper workstation/server operations, host power/GPU tuning, workload adaptation and quant fitting |
| [LM Studio](https://lmstudio.ai/) | Polished desktop and headless local-model experience, model discovery and developer ergonomics | Operator transparency and machine-level control |
| [Ollama](https://github.com/ollama/ollama) | Simplicity: install, pull, run; automatic scheduling and placement | Explainability and explicit control over how the machine is fitted and optimized |
| [KoboldCpp](https://github.com/LostRuins/koboldcpp) | Remarkable one-file portability and multimodal breadth | Multi-instance host management, optimization, fitting, GPU/power operation and machine forensics |
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI) | Generative-media graph authoring and ecosystem | Operating inference services and hardware, not authoring creative graphs |
| [vLLM](https://github.com/vllm-project/vllm) | High-throughput accelerator serving, tensor/pipeline/data/expert parallelism and multi-node production deployment | Bare-metal local/workstation optimization and small-fleet operations rather than datacenter scheduling |

The practical stack can therefore be compositional:

```text
people / agents / applications
        │
        ├── Open WebUI / another chat app
        ├── Hermes / Claude Code / MCP clients
        └── OpenAI-compatible clients
                    │
                    ▼
               LexiPanel
      access · gateway · MCP · fleet
                    │
      ┌─────────────┼──────────────┬─────────────┐
      ▼             ▼              ▼             ▼
  llama.cpp     sd.cpp         audio.cpp      ONNX/Camelid
      │             │              │             │
      └─────────────┴──────────────┴─────────────┘
                    │
                    ▼
       fit · benchmark · auto-fit
       VRAM/RAM · GPU · power · OS
```

---

## What it does

- **Independent instances.** Run multiple inference servers side by side, each with its own engine, devices, backend, parameters, port, logs, statistics and `systemd --user` unit (`LexiPanel-inst@<id>`). The original server remains available as the legacy `main` instance.

- **Five engine families, one operator workflow.** llama.cpp for text/vision; stable-diffusion.cpp for images; audio.cpp for speech, music and audio tasks; Camelid for curated GGUF chat on CPU/NVIDIA; ONNX Runtime GenAI for CPU/GPU/NPU execution.

- **220+ explained controls plus whatever the active build added yesterday.** Common settings live in grouped forms with plain-English help. For llama.cpp, *All build options* parses the active binary's `--help`, so a newly-added upstream flag does not require waiting for LexiPanel UI code before it is reachable.

- **Launch-plan preview.** See the exact command and environment, warnings, refused conditions and estimated memory before starting anything.

- **Memory planning.** Reads GGUF metadata for cache shape, uses observed process overhead, accounts for other servers already resident on the selected devices, exposes the host RAM budget and refuses launches that cannot fit safely.

- **Fallback tiers.** `normal` → `safe` → `minimal` after failed starts so a bad configuration degrades toward a known-good one instead of crash-looping the host.

- **Long-context measurement.** Measure decode at 8k, 32k, 128k, 240k or other depths with in-request thermals. Live traffic is compared against the measured curve at its own depth.

- **Optimizer.** Benchmark candidate launch settings against coding/agent workloads with thermal and abort limits. Models mode compares entire model files. Workload-weighted runs score the time a *typical request* would take rather than treating every depth equally.

- **Workload + auto-fit.** Keep real request evidence across restarts, learn idle windows, identify configuration/workload mismatches, measure alternatives, optionally apply narrowly-scoped speed wins and verify them on later real traffic.

- **Hardware-fitted requants.** Measure formats on the actual cards, solve a tensor-level plan to a real VRAM budget, build it on CPU, verify it against the current model, keep importance matrices and portable recipes, and watch upstream revisions.

- **Model-quality checks around optimization.** Task suites gate optimization/Fit comparisons. The refusal check separately measures expected refusals and over-refusals using public prompt sets, one ordinary chat request at a time.

- **Engine builds.** Browse upstream releases, install Vulkan/ROCm/CUDA/CPU variants, choose active builds per backend and optionally let the daily updater fetch new releases without restarting a running service.

- **Model management.** Resumable Hugging Face downloads with token support, instance references, protected deletion, chat-template extraction and linting.

- **Built-in llama.cpp Web UI.** LexiPanel can expose/configure the web UI shipped by llama.cpp per instance instead of reinventing its chat surface.

- **GPU visibility.** Per-card VRAM/GTT, process residency, thermals, clocks, PCIe state, `amdgpu_top`, power and power-cap history.

- **GPU tuning.** Supported AMD cards expose accepted OverDrive ranges for core/memory limits and undervolt, plus cap/performance/fan controls, benchmark comparison, tokens/joule, profiles, VBIOS backup and a ROM pre-flash check. LexiPanel never flashes firmware.

- **Power profiles.** CPU governor/idle policy, PCIe/NVMe power settings, GPU policy, fan/cap, watchdog and journal behavior can be grouped into audited profiles with a boot profile, drift detection, boot stability and PSU/UPS budgeting.

- **OpenAI-compatible gateway.** One `/v1/models` and `/v1/chat/completions` front door for running llama.cpp, ONNX Runtime and Camelid instances. Streaming passes through. Per-user rate/token/concurrency/model quotas and usage accounting are built in.

- **Access control.** Single-user mode for the home box, or multi-user mode with `viewer`, `operator` and `admin`, expiring API keys and a hash-chained audit log of mutating operations.

- **Fleet.** Run full LexiPanel on multiple machines. Members report hardware, instance state/speed and proposals to a primary. Shared models can appear through the primary gateway; replicas can be selected by open-request load and an unresponsive replica is skipped. Member control remains local.

- **MCP.** Expose the operator plane as tools to Claude Code, Claude Desktop, Hermes or any MCP client, using the same validation and access model as the UI/API.

- **Hermes Agent readiness.** For llama.cpp instances, check the settings Hermes needs, including Jinja/tool templates, context and network reachability, then generate the relevant `model:` and `mcp_servers:` configuration blocks without writing the LLM key into the config.

- **Crash forensics.** Boot history, clean-vs-hard-stop triage, settings active near a crash and a debug bundle designed to be readable by a human or AI assistant.

- **Protected file manager.** Stream multi-GB uploads, move/copy/rename/delete/download, zip folders and selections, while refusing access to hidden system directories and protecting files currently referenced by a server or setting.

- **Browser terminal.** ttyd behind the same front door, with a second Linux/PAM login before a shell exists.

- **Graph Gauntlet.** Yes, the charts can turn into a browser runner game whose terrain is the actual graph data. It makes no network requests and stops when hidden. This feature is not part of the control plane; it is here because staring at telemetry for months apparently has consequences.

---

## Tour of the tabs

| Tab | What you get |
|---|---|
| **Status** | All running servers, lifecycle controls, instance switcher, throughput/live decode, depth curve and profiles. Image/audio instances get task-specific run cards. llama.cpp instances get Hermes readiness/config. |
| **Parameters** | Grouped settings with tooltips, backend/device selection, memory calculator, RAM budget, fallback tiers and model profiles. ONNX instances get provider + generation controls. |
| **Optimize** | Coding/agent suites, candidate sweeps, model comparisons, workload-weighted runs, thermal/abort limits, apply/rollback. |
| **Workload** | Real traffic depth, prompt/output size, concurrency, draft acceptance, busy/idle heatmap, findings and Auto-fit experiments/proposals/history. |
| **Fleet** | Standalone/primary/member role; box health and reported hardware/instances on the primary; join/share controls on members. |
| **Access** | Mode, users, roles, API keys and the tamper-evident audit tail/chain check. |
| **Fit** | Source readiness, per-tensor planning, format speed measurements, builds, candidate report cards, importance matrices, portable recipes, conversion and refusal checks. |
| **Models** | Local models, sizes, references and protected delete. |
| **Download** | Hugging Face download queue/progress and token storage. |
| **Templates** | Extract/lint/pin chat templates for tool use and reasoning behavior. |
| **GPU** | Device memory/residency, thermals/power, clocks/link and cap. |
| **GPU Tuning** | Hardware details, supported tuning controls, benchmark/live power, profiles, VBIOS backup and ROM checks. |
| **Statistics** | Requests, tokens, decode/prefill history and draft acceptance. |
| **Diagnostics** | Pass/warn/fail checks with evidence, live command line, debug bundle and configuration backup. |
| **Power options** | Equipment, live/default/boot values, profiles, drift, audit log, boot stability and power budget/UPS. |
| **Crashes** | Boot history, hard-stop triage and the parameters in force. |
| **Builds** | Engine catalogs, releases, install/activate/delete and updater policy. |
| **Logs** | Engine/launch logs with follow. |
| **Files** | Protected home-folder manager with large streaming uploads, zip/download and keyboard operations. |
| **Terminal** | ttyd terminal and optional plain shell service. |

---

## Engines

| Engine | Server | Workloads | Compute | Configuration surface |
|---|---|---|---|---|
| **llama.cpp** | `llama-server` | Chat, coding/agents, vision, speculative decoding | Vulkan, ROCm, CUDA, CPU; multi-GPU through the supported llama.cpp path used by this setup | 164 first-class settings plus flags discovered from the active binary |
| **stable-diffusion.cpp** | `sd-server` | Text-to-image and supported diffusion models | Vulkan, ROCm, CPU | Model placement, offload, VAE behavior, generation defaults and presets |
| **audio.cpp** | `audiocpp_server` | TTS, cloning/design, STT, music/song, sound effects, separation, VAD, diarization | Vulkan, CPU | Server controls plus model-specific options generated from the installed build's specs |
| **Camelid** | `camelid serve` | Curated GGUF chat/agent use | CPU, CUDA/NVIDIA | Model, threads, cache precision, thinking/speculation, limits, auth/LAN |
| **ONNX Runtime GenAI** | `onnx_server.py` | OpenAI-compatible chat from ONNX GenAI models | CPU; AMD Ryzen AI/VitisAI; Intel OpenVINO NPU/GPU/CPU; Qualcomm QNN; CUDA; DML/WebGPU where the runtime supports them | Model/provider options plus generation options discovered/verified at start |

### audio.cpp at a glance

The installed audio.cpp build supplies the catalog. LexiPanel turns supported model families into install/run forms instead of hard-coding one TTS engine. That can cover TTS, cloning, ASR, music, sound effects, separation and diarization depending on the upstream build.

Residency controls let audio workloads share a machine with an LLM: maximum loaded models, idle unload and a keep-free memory guard.

### ONNX Runtime and NPUs

ONNX Runtime is how LexiPanel reaches devices llama.cpp does not target directly. An ONNX instance points at a GenAI model directory (`genai_config.json` + `model.onnx`) and a runtime Python/provider.

The panel reports the installed runtime/provider set and NPUs visible to the kernel. Starts are refused for missing runtimes/models/providers, missing QNN libraries and unsafe LAN listeners without an API key. The Parameters surface exposes the **17 generation options** currently reported by onnxruntime-genai and verifies them by setting/reading them at start rather than assuming the installed runtime accepts them. Provider-specific options cover OpenVINO, QNN and VitisAI plus an escape hatch for additional provider JSON.

CPU operation has been exercised in the source repository (the current README records Qwen2.5-0.5B-Instruct int4 on onnxruntime-genai 0.16 with all 17 options accepted/read back). AMD/Intel/Qualcomm NPU paths depend on the vendor runtime and real corresponding hardware, so treat a first run on those machines as hardware validation rather than a blanket compatibility guarantee.

---

## Architecture

```text
browser / API / MCP client
          │
          ▼
      Caddy :443
 TLS + front door
          │
          ├────────────▶ ttyd :8091 / :8092
          │
          ▼
    panel :8090
  (loopback only)
          │
          ├── JSON API / Access / audit
          ├── OpenAI gateway / quotas
          ├── MCP transport
          ├── fleet reporting/routing
          │
          ├──────── legacy main ────────▶ LexiPanel-llama.service
          │
          └──────── instances ──────────▶ LexiPanel-inst@<id>.service
                                           │
                                           ▼
                         llama-server / sd-server / audiocpp_server
                              / camelid / onnx_server.py
```

`panel.py` is the HTTP backend and orchestration center. Engine-specific modules live beside it. `instance_launch.py` executes the same launch plan the UI/API previewed, maintains failed-start state, archives logs and enforces the host-RAM watchdog.

Persistent state is deliberately boring: `.env` files, JSON/JSONL and per-instance directories. `make-backup.sh` captures the hard-to-reproduce configuration without copying model weights.

---

## Install

The tested path is Ubuntu 26.04 with a modern Python 3 (the project currently exercises Python 3.14), Caddy 2.x, ttyd and systemd. Python application code uses the standard library; `python3-jinja2` is optional for an extra template parse check. Individual inference engines keep their own native/runtime dependencies.

### Guided install

```bash
bash install-interactive.sh --check   # inspect account/paths; change nothing
bash install-interactive.sh           # guided install/update
```

Re-running the installer updates code while preserving instance/config/result state. The optional ONNX step can install the runtime environment; it is offered when an NPU is detected.

### Manual outline

1. Put the repository at `/home/admin/panel` and launch scripts at `/home/admin/llama`, or adapt the paths first.
2. Install Caddy/ttyd and the panel units:

```bash
bash install-panel-deps.sh
bash install.sh
```

3. Optional legacy `main` autostart:

```bash
bash install-autostart.sh
```

4. Allow user instances to survive logout:

```bash
sudo loginctl enable-linger admin
```

5. In **Builds**, install/activate an engine build for the backend you intend to use.
6. In **Status**, create an instance, choose engine/devices/backend, configure it under **Parameters**, inspect the launch plan and start it.

### Optional host features

- `sudo bash power/install-power.sh` installs the narrow root helper used by Power options/GPU Tuning. It changes no tuning value merely by being installed.
- AMD OverDrive clock/voltage controls require the included GRUB snippet and a reboot.
- `bash lockdown.sh` can restrict raw instance ports to a LAN subnet.
- audio.cpp families may need system libraries such as eSpeak NG.
- ROCm/CUDA/provider runtimes remain engine/vendor dependencies.
- `bash install-onnx.sh` creates the ONNX Runtime GenAI environment; provider-specific variants are available for supported runtimes.

---

## Parameters

The common llama.cpp surface contains controls for backend/offload, context/KV cache, RoPE/YaRN, batching/threads, speculative decoding, Vulkan/ROCm behavior, vision, reasoning, sampling, server settings and the host RAM floor.

The complete generated reference is **[PARAMETERS.md](PARAMETERS.md)**.

The important distinction is that parameters are not simply saved and trusted. Before launch, LexiPanel constructs the concrete plan and exposes:

- binary/build selected;
- exact argv;
- environment variables;
- devices/backends;
- estimated VRAM/RAM;
- warnings about slow/unsafe choices;
- hard refusal reasons.

`instance_launch.py` then runs that plan instead of rebuilding a second interpretation elsewhere.

---

## Workload and auto-fit

The optimizer asks **“what wins on a benchmark?”**. Workload asks **“what are people/agents actually doing to this instance?”**. Auto-fit connects the two.

For llama.cpp instances the profile records completed requests from the engine logs and preserves them across server restarts for **60 days**. It also aggregates already-existing slot telemetry into hourly activity and concurrency. LexiPanel's own optimizer, depth-curve, GPU-benchmark, refusal-check and Fit traffic is tagged out so the measuring does not teach the workload profile to measure itself.

Useful findings include:

| Evidence | What it can imply |
|---|---|
| Deepest conversations never use much of reserved context | Measure a smaller `CTX` and quantify the cache capacity recovered |
| Several requests repeatedly reach the limit | Measure a larger context if the estimator says it fits |
| Multiple parallel slots are configured but never used | Compare against `PARALLEL=1` |
| All slots spend meaningful time busy | Report a capacity pressure condition rather than blindly shrinking concurrency |
| Draft acceptance is consistently poor | Measure speculation-off against current behavior |
| Real decode falls well below the depth curve | Look for throttling, contention, build/config drift or other environmental change |
| Post-change real traffic is slower/faster at matched depths | Confirm or roll back the change |

Auto-fit starts only under its configured safety conditions: instance healthy, no competing measurement/restart, quiet period satisfied, an allowed idle window with enough time left, no unverified previous change and weekly experiment limits.

A real request arriving interrupts the experiment and returns the instance to its saved settings after that request. Capacity-changing decisions such as context/parallelism remain proposals. Automatic application is restricted to an allowlist of speed-oriented keys (`UBATCH`, `BATCH`, `CACHE_REUSE`, `SPEC_N_MAX`, `SPEC_P_MIN`, `THREADS`) and requires at least the configured **3%** typical-request gain plus non-regressed task quality. Changes to capacity or behavior such as context, slot count, or speculation on/off remain proposals.

After application, later real requests are compared with up to 14 days of pre-change traffic at matched depth bands. At **20 matched requests**, under **93%** of prior matched-depth decode is a regression: auto mode rolls it back and will not auto-apply that change again on that configuration. Too little traffic after seven days is inconclusive. A rollback leaves alone settings changed by hand since the experiment.

Run the included stand-in end-to-end test to watch the logic without a GPU or your real configuration:

```bash
bash tests/e2e/run.sh confirm
```

---

## Fit: hardware-fitted requants

Fit treats quantization as a constraint/measurement problem rather than a filename choice.

1. **Source** — pin/download a revision, hash it and convert to a full-precision/BF16 GGUF source.
2. **Measure formats** — create representative quant test files and benchmark them on the actual cards.
3. **Plan** — derive a byte budget from a configuration already known to fit; combine measured format speed with tensor-role sensitivity and component-specific handling.
4. **Dry-run** — ask the quantizer for the exact predicted output size and confirm that overrides land where intended.
5. **Build** — quantize at low CPU priority so the serving stack can remain useful.
6. **Verify** — compare candidate models with task suites, depth curves and observed VRAM.
7. **Parts/recipes** — handle output/embedding/MTP components, importance matrices and portable/explainable recipes; convert supported projectors/sd.cpp/audio.cpp artifacts.
8. **Watch upstream** — report source/quantizer revision changes without silently replacing a production model.

Quality planning is explicit about its evidence. Where empirical KL data is not available, the planner uses priors and says so rather than presenting an estimate as measured truth.

See **[fit/README.md](fit/README.md)** and **[fit/FINDINGS.md](fit/FINDINGS.md)**.

---

## GPU Tuning

The GPU Tuning tab answers a deliberately narrow question:

> **What is this card worth on this model, and what did this change buy?**

It exposes card identity/VBIOS/link state, DPM tables, sensors and accepted tuning ranges. A benchmark drives the *running model* while sampling power, clocks, temperatures and load.

Results include decode, cold prefill, run-to-run spread, power, peak thermals and **tokens per joule**. Results keep the settings that produced them and can be compared against a reference.

On supported AMD hardware, **Apply and test** changes values, runs the benchmark and restores the prior values if the run fails, the model server dies, a reset is seen or a thermal limit trips. Proven profiles can later become the boot profile applied before inference services start.

VBIOS support is intentionally read/check-only: back up the image, inspect a candidate ROM and print the vendor utility command. LexiPanel does **not** flash firmware.

---

## Gateway and quotas

The panel exposes one OpenAI-compatible front door for the supported chat engines:

```text
GET  /v1/models
POST /v1/chat/completions
```

A requested `model` resolves to an instance id/alias/model name. Streaming is passed through.

In multi-user mode the caller uses a LexiPanel API key while per-instance credentials stay behind the gateway. Admins can configure per-user requests/minute, tokens/day, concurrency and allowed models. Usage is counted per day/user/model; prompt/reply contents are not stored by gateway accounting.

Fleet members can optionally share eligible running models to the primary. Multiple copies of the same model can act as replicas; requests prefer the replica with fewer open requests and an unresponsive replica is temporarily skipped.

This is intentionally a local/small-fleet gateway, not a claim to replace mature cloud-provider routing stacks.

---

## Access: single-user or multi-user

**Single-user** is the default home-box mode: Caddy's login protects the panel and the app trusts traffic that reaches it.

**Multi-user** makes the panel authenticate every request itself, including loopback clients:

| Role | Intended access |
|---|---|
| **viewer** | Read-only operational state |
| **operator** | Viewer + lifecycle and measurement operations such as start/stop/restart, benchmark/optimizer/depth/auto-fit experiments |
| **admin** | Configuration, access, files, builds, fleet, power/GPU tuning and other mutations |

Passwords are stored through the project's password-hashing path; API keys are shown once, expire and cannot exceed their user's role. MCP calls inherit the caller's role.

Mutating actions enter a hash-chained audit log. Tests include access-control tripwires so adding a route without classifying it cannot silently turn it into viewer access.

---

## Fleet: many boxes, one primary

Each box runs a full LexiPanel. One can be **primary** and others **members**.

A member joins with a short-lived one-time code and receives its own token. It reports outward to the primary roughly once per minute. Reports contain machine/instance operational metadata such as GPUs/NPUs/RAM, engine/model/state/speed, workload counts, pending Auto-fit proposals and LexiPanel version — not prompts, instance secrets or API keys.

The primary marks boxes online/stale/offline and can revoke a member. Normal control remains on each member's own panel; the fleet layer is not pretending to be a distributed shell.

The shared-model gateway adds a useful middle ground before full distributed inference: one client endpoint can see models resident across several boxes and use simple replica failover/load selection.

See **[docs/FLEET.md](docs/FLEET.md)** for the design and future work such as llama.cpp RPC striping.

---

## MCP server

`mcp_server.py` exposes the panel as Model Context Protocol tools through stdio or Streamable HTTP.

```bash
# stdio over SSH
claude mcp add lexipanel -- ssh admin@box python3 /home/admin/panel/mcp_server.py

# HTTP through the panel front door
claude mcp add --transport http lexipanel https://box/api/mcp \
  --header "Authorization: Basic $(printf 'admin:PASSWORD' | base64)"
```

`LEXIPANEL_MCP_READONLY=1` strips mutating tools. Multi-user API keys can be used instead of the single-user front-door credential where configured.

The tool surface is deliberately operational: **28 MCP tools** covering status, instances, parameters and memory estimates, launch plans, models, GPU/power, benchmarks, curves, optimizer/workload/Auto-fit state, diagnostics, crash/log information and controlled text-file reads.

### Hermes Agent

For [Hermes Agent](https://hermes-agent.nousresearch.com/), LexiPanel checks a llama.cpp instance for Jinja/tool-template readiness, usable context per conversation and network reachability, then emits the relevant `model:` and `mcp_servers:` config fragments.

The generated MCP config starts read-only. An LLM server API key is referenced through an environment variable rather than written as a literal secret.

---

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
browser, not on the GPU box. Your record, bests and achievements are kept in the browser. Sources,
the installer and tests are in [`gg/`](gg/). [`gg/GG-AI-GUIDE.md`](gg/GG-AI-GUIDE.md) explains
how to rebuild or extend it without breaking the panel.

---

## HTTP API

The web UI is intentionally thin over a JSON API. The panel itself binds loopback (`127.0.0.1:8090`) and is intended to sit behind Caddy.

The generated **[API.md](API.md)** is the authoritative route list and currently documents **173 routes** covering instance lifecycle/configuration, optimization, workload/Auto-fit, Fit, models/downloads/templates, engines/builds, image/audio/Camelid/ONNX functions, files, GPU/power/tuning, access, gateway, fleet, Hermes, MCP, diagnostics/crashes/logs and backups.

Examples:

```bash
# running servers
curl -s http://127.0.0.1:8090/api/servers | jq

# exact launch argv before a start
curl -s 'http://127.0.0.1:8090/api/launch-plan?inst=coder' | jq '.argv, .errors, .warnings'

# gateway model catalog
curl -s https://box/v1/models -H 'Authorization: Bearer lp_...'
```

Browser cross-site protections intentionally reject request shapes that can be forged by another site while a browser automatically reuses credentials. Scripts should send JSON for POST bodies.

---

## Safety rails (learned the hard way)

These are part of the implementation rather than recommendations in a wiki:

- **No accidental second full model as a draft.** A target model used as its own external speculative draft can load another full copy; the planner refuses the known-dangerous case.
- **No silent Vulkan system-memory fallback.** Launch plans force `GGML_VK_ALLOW_SYSMEM_FALLBACK=0` so an allocation failure does not quietly become a catastrophically slow host-RAM run.
- **No accidental iGPU choice on the Linux/Vulkan path.** Device selection/pinning is explicit; CPU audio instances hide Vulkan ICDs.
- **Host RAM floor.** The launcher can kill an instance before `MemAvailable` crosses the configured floor, and planning can refuse impossible starts up front.
- **Fallback tiers.** Failed starts move toward safer known configurations instead of repeatedly executing the same broken command.
- **Configuration refusal is not a service crash.** A refused plan uses a distinct exit path so systemd does not pointlessly retry it.
- **Main start guard.** Repeated failed starts and restarts during loading are bounded and surfaced with the reason and reset path.
- **Measurement jobs do not pile on top of each other.** Optimizer/depth/GPU/Fit operations use coordination so competing benchmark jobs do not make each other's numbers meaningless.
- **GPU tuning is reversible.** Unstable live tuning can be restored, and persistent tuning requires an explicit boot-profile choice.
- **No firmware writer.** ROM support stops at backup/check/instructions.

---

## Security

LexiPanel is a machine-control application, so the useful security question is not *“does it have a login?”*. It is *“what can a browser, API client, agent or compromised model actually cause the host to do?”*

Current guard rails include:

- panel listener bound to loopback, with Caddy as the intended front door;
- cross-site request checks appropriate to browser credential behavior;
- literal/single-quoted settings serialization with newline rejection for shell-sourced env files;
- raw inference ports expected to remain loopback unless deliberately exposed;
- optional UFW LAN restriction helper;
- terminal protected by the front door **and** PAM/Linux account login;
- narrow sudoers/root-helper verbs rather than arbitrary root shell execution;
- hardware-target and accepted-range validation in the power helper;
- protected file-manager roots and in-use/reference checks;
- MCP read-only mode and multi-user role inheritance;
- API keys with scope-by-role/expiry;
- hash-chained change audit;
- safety tests that trip if code begins invoking firmware flashers, writes privileged sysfs outside the helper or broadens sudoers unexpectedly.

This is still self-hosted software with a browser terminal and privileged optional hardware controls. Read the code and threat model before exposing it beyond the network/users you intend.

---

## macOS (experimental)

`hostos.py` abstracts the operating-system questions LexiPanel asks, including memory/process/listener/service-manager data. Linux/systemd is the exercised path. A launchd/macOS path exists, but Linux-specific features such as AMD sysfs tuning, DRM residency and journald crash triage naturally report unavailable.

Treat macOS as experimental until exercised and reported by actual Apple Silicon users.

## Windows (experimental)

`hostos.py` answers memory, process, listening-port, and NVIDIA GPU questions on Windows. `panel.py` no longer imports `fcntl` at startup, so `python panel.py` can serve the page. Instances are started with the current Python, not a systemd unit. Sysfs clocks, AMD power caps, journald crash triage, and the DRM residency path report nothing on Windows, the same way they report nothing on macOS.

The panel still binds to localhost. Point a browser at the address it prints. GPU tuning and power profiles remain Linux features.

---

## Adapting it to your box

The remaining “real machine” assumptions are intentionally documented rather than hidden behind a generic installer:

- **User/paths:** the original layout is `admin` under `/home/admin/{panel,llama,models,sdcpp,audiocpp,...}`. Search the repo for `/home/admin` before deploying under another account/layout.
- **Legacy `main`:** the historical launch scripts were written around the original Qwen/AMD setup. You can repoint them or ignore `main` and use ordinary instances.
- **Hardware:** AMD is the deepest Linux implementation because that is the machine the project came from. NVIDIA is supported through its available management/runtime paths. Multi-GPU capabilities follow the underlying engine/backend rather than pretending every combination is equivalent.
- **NPUs:** provider detection/configuration exists, but a provider only works when its vendor runtime and compatible hardware/model are actually present.
- **Ports:** the panel/terminal/main/telemetry/instance defaults are local conventions, not protocol requirements; edit them if they collide with your environment.

---

## Configuration files

Important state remains plain-file based:

| State | Purpose |
|---|---|
| `params.env`, backend/tier env files | Legacy main + fallback configuration |
| `instances/<id>/instance.json` + `params.env` | Per-instance identity and launch configuration |
| engine/build state | Active builds and updater policy |
| `profiles/`, `curves/`, `optimize/` | Model profiles and measurement history |
| `workload/<id>/` | Preserved request/activity/config/Auto-fit evidence |
| power/GPU tuning state | Profiles, audit, benchmarks and VBIOS backups |
| access/gateway/fleet state | Users/keys/audit, quotas/usage and member/primary metadata |

Back up the irreplaceable configuration with `make-backup.sh`; model weights are intentionally not included.

---

## Checks before an upload

The repository includes unit/fake-sysfs tests, Auto-fit end-to-end tests and a combined pre-upload runner.

```bash
bash tests/run_all.sh          # full suite used by CI
bash tests/run_all.sh --quick  # compile/unit/script checks
```

Browser tests require Node/Playwright setup. Hardware-independent tests should not need a GPU.

`tests/test_safety.py` is deliberately a tripwire for actions the project never wants to learn how to do casually, including firmware flashing, broad privileged sysfs writes and widened sudo permissions.

Generated docs such as `PARAMETERS.md` and `API.md` should be regenerated/checked with code changes so claims do not drift from the implementation.

---

## File layout

```text
panel.py               HTTP backend: routes, orchestration, launch plans, estimators, diagnostics
instance_launch.py     execute a plan; failure counter, RAM watchdog, log archival

auth.py                single/multi-user auth, roles, API keys, hash-chained audit
gateway.py             unified /v1 routing, per-user quotas and usage
fleet.py               primary/member join, reports and fleet model routing
docs/FLEET.md          fleet behavior and future distributed work

engines.py             upstream engine catalogs/build activation/update policy
sdcpp.py               stable-diffusion.cpp instances and generation
camelid.py             Camelid instances/catalog/chat proxy
audiocpp.py            audio.cpp catalog, instances, runs and outputs
onnxrt.py               ONNX engine configuration/provider/NPU detection
onnx_server.py          OpenAI-compatible ONNX Runtime GenAI server
install-onnx.sh         ONNX runtime environment installer

webui.py               llama.cpp built-in web UI integration
hermes.py              Hermes readiness checks/config generation
mcp_server.py          stdio + Streamable HTTP MCP tools

optimizer.py           candidate/model benchmarking and workload-weighted optimization
optimize_suite.py      coding/agent benchmark tasks
depthcurve.py          decode-vs-context measurement with thermals
workload.py            real-request profile, activity envelope and findings
autofit.py             idle experiments, proposal/apply, verification and rollback

fitquant.py            Fit measurement/solver/build/verify jobs
fitrecipe.py           portable/explainable tensor-role recipes
fit/                   source/measurement docs and worked findings
refusals.py            expected-refusal / over-refusal evaluation

gpupower.py            per-GPU caps
gputune.py             GPU detail/tuning/benchmark/VBIOS checks
poweropts.py           host power profiles, drift, stability and budgets
power/                  constrained privileged helper, sudoers and boot unit
hostos.py               Linux/macOS operating-system abstraction

filemgr.py             protected home-folder file manager
flagcatalog.py          discover active-build flags from --help
flaghelp.py             plain-English help
main_devices.py         legacy-main device selection

static/index.html       entire browser UI; no build step
gg/                     Graph Gauntlet source/tests/guide
systemd/                services, front-door templates, sudoers/OverDrive snippets
launch-scripts/         legacy main scripts and backend benchmark helper
examples/               example configuration
tests/                  unit, safety, UI and E2E checks
.github/workflows/      repository CI

install-interactive.sh  guided install/update/check
install-panel-deps.sh   host dependencies and optional runtime packages
install.sh              Caddy + panel + ttyd setup
install-autostart.sh    legacy-main boot service
lockdown.sh             LAN firewall helper
fix-firewall.sh         firewall change with rollback guard
make-backup.sh          configuration/state backup

audit-flags.sh          compare launch flags with the active binary
PARAMETERS.md           generated parameter reference
API.md                  generated HTTP API reference
CHANGELOG.md            release history
```

---

## What LexiPanel is not

A broad control plane is easy to oversell, so the boundary matters:

- It is **not** a replacement for Open WebUI's chat/RAG/collaboration product. Use Open WebUI in front if that is the experience you want.
- It is **not** a replacement for llama.cpp; llama.cpp is one of the engines that makes the project useful.
- It is **not** as frictionless as Ollama or LM Studio for a person who simply wants to download one model and chat.
- It is **not** as polished or cross-platform a guided local-AI/browser product as LumaBrowser.
- It is **not** as compact a one-file multimodal package as KoboldCpp.
- It is **not** as generic a hot-swap proxy as llama-swap today.
- It is **not** as broad a backend ecosystem as LocalAI.
- It is **not** a ComfyUI-style media workflow graph.
- It is **not** a vLLM/Kubernetes/Ray-scale distributed inference scheduler.
- It is **not** fully hardware-agnostic yet. AMD/Linux remains the deepest path and NPU/macOS support needs more real machines.
- It is **not** mature merely because a feature exists in source. This is a young project and needs more external hardware, users, soak time and reproducible comparative data.

Those are design boundaries and maturity limits, not hidden footnotes.

---

## Why keep building this instead of gluing tools together?

You absolutely can glue together a server runner, GPU monitor, benchmark scripts, quantizer, reverse proxy, auth layer, Prometheus dashboards and a collection of shell helpers. LexiPanel started that way.

The value of combining them is that the pieces can share evidence:

- the memory estimator knows what every other managed instance already occupies;
- the optimizer knows which depth bands real requests spend their decode time in;
- Auto-fit knows when the box is normally idle and whether a change survived later traffic;
- Fit knows the VRAM budget of a configuration that already works and the quant formats measured on those exact cards;
- GPU tuning knows the throughput and power of the model actually served;
- crash triage can connect a failed boot with the settings/power profile that were active;
- the gateway/fleet knows which managed instances are actually available;
- an MCP client can ask for that evidence through the same rules the browser uses.

The integration is the feature.

---

## Roadmap direction

The project is already broad enough. The highest-value work is increasingly about **maturity and generalization rather than adding random tabs**:

- exercise the existing paths across more AMD, NVIDIA, Intel, Apple and NPU machines;
- remove historical `/home/admin` assumptions from installation/state paths;
- deepen engine/plugin abstraction so a new server can be added without teaching every panel subsystem about it;
- add stronger optimizer search strategies while preserving workload-aware scoring and real-traffic verification;
- make fleet routing/health more observable and continue the documented distributed/RPC experiments;
- keep generated docs, tests and security tripwires synchronized with the rapidly growing route/feature surface;
- publish reproducible hardware/model benchmark and Fit case studies rather than generic performance claims.

See **[docs/ROADMAP.md](docs/ROADMAP.md)** for the implementation roadmap.

---

## Credits

LexiPanel stands on upstream projects rather than hiding them behind a brand. In particular:

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- [audio.cpp](https://github.com/0xShug0/audio.cpp)
- [ggml](https://github.com/ggml-org/ggml)
- [Caddy](https://caddyserver.com/)
- [ttyd](https://github.com/tsl0922/ttyd)
- [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai)

The comparative projects linked earlier are also worth reading directly. Many are better at their own layer; LexiPanel's purpose is to make those layers easier to operate as one measurable local-AI machine.

Model weights, runtimes and upstream components retain their own licenses.

## License

[MIT](LICENSE) © 2026 W61k3r
