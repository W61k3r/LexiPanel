# Fit: hardware-fitted requants

**Goal:** start from a model's full-precision (BF16) source and choose each tensor's format so
the model fits your cards at the context and speed you need, **without losing agentic or coding
quality**, using the formats those cards actually run fastest.

Fit is a LexiPanel tab (`fitquant.py`, routes under `/api/fit/`, see `API.md`). The two scripts
here are phase A and the original phase B; the tab runs both as jobs too. Read
**[FINDINGS.md](FINDINGS.md)** first: the worked example on Qwen3.8-27B / 7900 XTX (model
layout, where a published quant spends its bits, KV-cache math, conversion gotchas, measured
format speeds, the solver's first plans).

## Phases

| Phase | Where | What it does | Touches the server? | Status |
|---|---|---|---|---|
| A. Source | `phaseA-source.sh`, or Fit tab → Upstream watch → Fetch | Download the pinned safetensors, check every file's SHA-256, convert to a BF16 GGUF, check the MTP head | no | done and verified on the example |
| B. Formats | Fit tab → Measure formats (or `phaseB-types.sh`) | One single-format test file per format, llama-bench on each card of an instance, a per-card speed model | stops the instance, always restarts it | measured on the 7900 XTX |
| C. Plan | Fit tab → Plan | Per-tensor formats inside the measured VRAM budget; explains every choice; exact sizes from `llama-quantize --dry-run` | no | done |
| D. Build / verify | Fit tab → Build, Candidates → Verify | Low-priority CPU quantize; the optimizer's models mode compares candidates with today's model (suites, depth curve, real VRAM) | verify restarts the server once per model | built and exact; first verify pending |
| E. Parts | Plan (output / MTP / embeddings), Parts and other engines | Component precision; projector, sd.cpp and audio.cpp conversion | no | done |
| F. Watch | Upstream watch | Daily check for a newer source revision or quantizer; reports only | no | done |
| KLD | not yet | KL divergence against a Q8_0 reference, to replace the priors | yes | not started |

Where things go: sources in `~/models/src/<name>/`, format test files in `~/models/fit-test/`,
candidates in `~/models/fit/` (each with a `.fit.json` sidecar: plan, source revision, quantizer
build), and the panel's Fit state (card profiles, plans, job logs, importance matrices) in
`<panel dir>/fit/`, next to these files.

## Using the scripts

```bash
bash fit/phaseA-source.sh --check     # preflight only; changes nothing
bash fit/phaseA-source.sh             # asks once, then runs; safe to stop and re-run (it resumes)

bash fit/phaseB-types.sh quantize     # CPU only; your server keeps serving
bash fit/phaseB-types.sh bench        # STOPS the server, benchmarks, restarts it (trap)
bash fit/phaseB-types.sh deep         # the 3 fastest formats at 128k depth
bash fit/phaseB-types.sh report       # results table
bash fit/phaseB-types.sh clean        # delete the test quants (asks)
```

- **Phase A defaults** to the worked example: the repo, a pinned revision, and the llama.cpp
  converter tag. Override them with `REPO=… REV=… NAME=…`.
- Tools go in `~/fitquant/`: `uv`, a private Python 3.12, and CPU-only PyTorch. No sudo.
- Sources go in `~/models/src/<NAME>/`.
- **Phase B** stops and starts the `LexiPanel-llama` unit with the passwordless sudoers rule that
  `install-autostart.sh` installs. It mirrors the server's GPU settings (Vulkan ICD pinning,
  sysmem fallback off, flash attention, iq4_nl KV).

## Needs

- A llama.cpp CPU build (for `llama-quantize`) and a GPU build (for `llama-bench`) under
  `~/llama/`, as the Builds tab installs them.
- Disk: the source (~56 GB for a 27B) + its BF16 GGUF (~55 GB), plus one test file per
  format while measuring (the tab's Measure job works in rounds and deletes them when the disk
  is short) and 14-20 GB per candidate. Fit always keeps 20 GB free.
- The GPU build's `llama-bench` (Measure formats) and `llama-imatrix` (Importance matrix).
- **Converting recent Qwen uploads needs transformers 5.** llama.cpp still pins 4.57.6;
  phase A handles this (see FINDINGS §5).
