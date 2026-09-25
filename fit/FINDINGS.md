# Fit findings: building hardware-fitted GGUFs

Worked example: **Qwen3.8-27B (DavidAU TurboFCFusion MTP) on a Radeon 7900 XTX, 24 GB**,
as LexiPanel's Fit pipeline was being built (2026-09-23/24). Everything here was measured or
checked on that box unless it says otherwise. The method carries over to other models and
cards; the numbers don't.

---

## 1. Ground rules (read first)

- **Always start from the full-precision source** (safetensors or a BF16 GGUF), never a
  requant of someone's Q4. Requantizing an already-quantized file stacks rounding error;
  `llama-quantize` warns it "can severely reduce quality". A stock quant can only ever go
  **down** in quality.
- **CPU jobs are safe while your server keeps serving:** conversion, quantizing, and dry runs run
  alongside it. Keep them at `nice -n 19 ionice -c3`. RAM is the shared limit: watch
  MemAvailable. The test box has 30 GB, and its main server uses about 11 GB of process RAM.
- **GPU jobs need the server stopped** when it fills the card. The XTX had about 160 MiB free
  while main ran (VRAM 24,397 / 24,560 MiB). Stopping it also drops its cached conversation, so
  an agent's next turn re-prefills for minutes. **Never let an agent stop the server its own
  model runs on.**
- **Never give `--spec-draft-model` a full-size model.** It hard-locked this host twice.
  MTP is embedded (`--spec-type draft-mtp`).

## 2. The model: Qwen3.8-27B TurboFCFusion (DavidAU)

| | |
|---|---|
| Source | `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU`, pinned revision `fd6a268`, 55.6 GB safetensors: 12 shards plus `model-mtp-restored.safetensors` |
| Architecture | `Qwen3_5ForConditionalGeneration`, GGUF arch `qwen35`, hybrid |
| Layers | 64 plus 1 MTP layer (block 64). **Only 16 of the 64 are full attention**; the other 48 are linear attention (`ssm_*` tensors, fixed-size state) |
| Attention | 24 query heads, **4 KV heads**, head_dim 256, hidden 5120, FFN 17,408 |
| Vocabulary | 248,320 tokens. Embedding and output are each about 1.27 B parameters (2.4 GiB each at BF16) |
| BF16 GGUF | `~/models/src/Qwen3.8-27B-TurboFCFusion-735-882/Qwen3.8-27B-TurboFCFusion-735-882-BF16.gguf`, 54.6 GB, 866 tensors (506 BF16, 360 F32), SHA-256 in the `.sha256` beside it. `manifest.json` holds the hash of every source file |
| MTP head | Complete: a full block 64 (attention, FFN) plus `nextn.eh_proj`, `enorm`, `hnorm`, `shared_head_norm` |
| Tokenizer | **Identical** to the published GGUF's: all 248,320 tokens, 247,587 merges, eos 248046, pad 248044, same 8,953-char chat template (hash-compared) |

### KV cache cost: why context is cheap on this model

Only the 16 full-attention layers keep a KV cache. Per token: 2 × 4 heads × 256 × 16 layers
= 32,768 values.

| KV type | per token | 240k ctx | 200k ctx | saved by 240k → 200k |
|---|---|---|---|---|
| iq4_nl (main today) | 18.0 KiB | 4.12 GiB | 3.43 GiB | **0.69 GiB** |
| q8_0 | 34.0 KiB | 7.79 GiB | 6.48 GiB | 1.30 GiB |
| f16 | 64.0 KiB | 14.66 GiB | 12.21 GiB | 2.45 GiB |

Dropping context buys little VRAM here. Raising KV precision costs a lot.

## 3. DavidAU's recipe (the MTP Q4_K_M main runs today)

17,631 MiB of weights. Where the bits go:

| Tensor group | Type | Notes |
|---|---|---|
| `output.weight` | **BF16, 2,425 MiB** | the "MAX" in the name. A standard Q4_K_M puts it at Q6_K (about 0.8 GiB) |
| `token_embd.weight` | Q4_K, 682 MiB | |
| MTP block 64 (all matrices) | Q8_0, about 430 MiB | protects draft acceptance |
| `ffn_down` | half Q6_K (32), half Q4_K (32) | standard Q4_K_M mix |
| `attn_v` | half Q6_K, half Q4_K | |
| `attn_qkv` (linear-attention layers) | half Q6_K, half Q4_K | |
| everything else (`ffn_gate/up`, `attn_q/k/output`, `attn_gate`, `ssm_*`) | Q4_K | `ssm_a`, `ssm_conv1d` and `ssm_dt` stay F32 |

A **standard Q4_K_M from the same BF16 is 16,021 MiB (16.8 GB)**, about 1.6 GiB smaller,
almost all of it the output layer. **The first open question for the fit:** does 2.4 GiB on
the output layer beat spending those bits in the middle layers (ffn_down, attention),
measured on agentic and coding work?

## 4. The target envelope on the test box

The current setup is a known-good fit: 240,128 ctx, KV iq4_nl, MTP on,
mmproj on, VRAM 24,397 / 24,560 MiB. A new quant must fit **the same envelope** and either
match quality while being faster, or be better at the same speed. Dropping to 200k is
acceptable **only** for a significant gain, and on this model it frees just 0.69 GiB (section 2).
**Agentic and coding quality must not drop.**

Measured decode curve, for reference: 58.9 / 47.5 / 34.4 / 25.5 t/s at 8k / 32k / 128k /
240k. That was measured for an earlier model; re-measure with the depth curve, since one t/s
number misleads.

## 5. Conversion gotchas (solved)

- **transformers 5 tokenizers.** Uploads saved with transformers 5 name their tokenizer
  class `TokenizersBackend`. llama.cpp b11011 **and** b11149 pin transformers 4.57.6, which
  can't load it: the conversion dies at "Set model tokenizer" with
  `ValueError: Tokenizer class TokenizersBackend does not exist`. **Fix:** install
  transformers 5 (5.17.0 works) instead of the requirements file. The resulting vocabulary
  was verified identical to the published GGUF (section 2).
- The b11011 converter is a package now (`convert_hf_to_gguf.py` plus `conversion/`). Qwen
  3.5/3.6/3.8 MTP is handled by `_QwenMtpMixin` and kept by default (`--no-mtp` drops it,
  `--mtp` exports it alone).
- The prebuilt llama.cpp releases don't ship the converter. Clone the matching tag.
- Python 3.14 on this box has no pip. Use `uv` with a private Python 3.12 in `~/fitquant`
  (no sudo). PyTorch is the CPU-only wheel.
- The converted file is `general.file_type = 32` (BF16), about 16 BPW.

## 6. Tools

- `fit/phaseA-source.sh`: download (pinned, resumable), SHA-256 verify, BF16
  convert, inspect, dry-run size check. `--check` only runs the preflight.
- `llama-quantize` (b11149) supports:
  - `--tensor-type NAME=TYPE` and `--tensor-type-file` to set the format per tensor
  - `--imatrix`
  - `--dry-run` for the size without writing
  - `--output-tensor-type` and `--token-embedding-type`
  - `--leave-output-tensor`
- ggml types include MXFP4 and NVFP4. The only FP4 preset is `MXFP4_MOE`.
- Other tools: `llama-imatrix`, `llama-perplexity` (KL divergence against a reference),
  `llama-bench`, `llama-fit-params`.
- The panel's decode-vs-depth curve and optimizer suites (agentic, coding) are the quality
  and speed gates.
- **No importance matrix is published** for this model. DavidAU's NEO/CODER imatrix is
  private. Computing one needs the model running, so it's GPU work.

## 7. Hardware formats (what "optimal integer format" means per card)

- **7900 XTX (RDNA3):** WMMA is rated 246 TOPS INT4 and 123 TOPS INT8, **but llama.cpp
  does 4-bit weights with 8-bit activations, i.e. the INT8 path.** 4-bit wins on the XTX
  because of **memory bandwidth** (960 GB/s): decode is bandwidth-bound. It has no FP8 or
  FP4 hardware, so FP4 formats only save memory. Try first: Q4_0, IQ4_NL, Q4_K, IQ4_XS,
  with Q6_K and Q8_0 for sensitive tensors. **Measure; don't assume** (phase B).
- **RTX 2060 (Turing):** INT8 tensor cores. The k-quants and IQ types run through int8
  kernels.
- **Other vendors** (from published sources, not measured here):
  - NVIDIA Blackwell has native NVFP4 and MXFP4 (llama.cpp path merged April 2026).
  - RDNA4 has FP8 and INT8 WMMA; several k-quants are slower there.
  - Intel Arc Xe2 does best on Q4_0 and Q8_0 ("reorder" kernels); it has no FP8.

## 8. Phase B results: format speed on the 7900 XTX (measured 2026-09-24)

Single-format ("--pure") 27B files, llama-bench b11149 Vulkan, FA on, KV iq4_nl, b2048/ub512,
2 runs each, server stopped. Raw decode, no MTP.

| format | GiB | pp512 @0 | tg128 @0 | pp512 @32k | tg128 @32k |
|---|---|---|---|---|---|
| IQ4_XS | 13.53 | 909.3 | **40.41** | 484.7 | **36.11** |
| Q4_K | 14.32 | 874.6 | 38.53 | 491.3 | 34.96 |
| IQ4_NL | 14.32 | 885.4 | 38.53 | 495.1 | 34.02 |
| Q4_0 | 14.32 | **980.9** | 38.20 | **518.9** | 34.71 |
| Q5_K | 17.50 | 891.1 | 33.58 | 496.4 | 30.96 |
| Q6_K | 20.88 | 815.6 | 30.16 | 468.7 | 27.76 |

- **IQ4_XS is both the smallest and the fastest decoder.** Q4_0 prefills fastest; nothing else
  about it wins.
- Decode fits `time/token = 9.9 ms + GPU bytes / bandwidth` with effective bandwidth 838-867 GiB/s
  for every format (the card's peak is 894 GiB/s). So on this card decode is almost purely a
  function of bytes read; the format barely matters beyond its size.
- Consequence: DavidAU's BF16 output layer (2,425 MiB, read every token) costs about 2.7 ms/token,
  roughly 8% of decode, compared with Q6_K (995 MiB).

## 9. Phase C: the solver (the Fit tab, `fitquant.py`)

- **Sizes are exact:** `llama-quantize --dry-run --pure <src> <TYPE>` lists every tensor's size in
  that format in ~0.2 s (header only), including k-quant fallbacks. The solver's totals match the
  final dry run within 2 MiB.
- **`--tensor-type` patterns are regex SEARCH and case-insensitive.** `output.weight=q6_k` also hits
  every `attn_output.weight`. Always anchor: `^blk\.3\.ffn_down\.weight$=q8_0`. Overrides are
  honoured with `--pure`.
- **token_embd is CPU-resident** in llama.cpp (input layer), so its format costs host RAM only:
  no VRAM and no decode time. The solver puts it at Q8_0.
- **Budget = today's GPU weights + VRAM measured free now - margin.** The running config is known to
  fit, so this is calibrated rather than modelled. On the test box (2026-09-24): 16,949 MiB weights +
  1,646 MiB free - 512 margin = 18,083 MiB.
- **Quality ranking is a PRIOR** (per-tensor sensitivity: output 8x, MTP 3x, attn_v 3x, linear-attn
  QKV 2.5x, ffn_down 2x, first/last eighth of layers 1.4x; error ~ 4^-bits). KL divergence is
  not measured yet; candidates are verified on the agentic/coding suites instead.
- First plans against the running DavidAU Q4_K_M (predicted 34.1 t/s raw):
  - "faster": same prior quality, 37.3 t/s (+9%), 14.5 GiB GPU weights (2.1 GiB less)
  - "better": same speed, prior loss about halved (index 0.49)
  - "max": fills the budget, 32.5 t/s, index 0.27
  - The priors keep the output layer at Q6_K: per byte, bits in ffn_down / attn_v / QKV buy more.
    Whether DavidAU's 2.4 GiB BF16 output beats that is exactly what verification answers.

## 10. Importance matrix

`llama-imatrix` (in the Vulkan build) with `-fit on -fitt 512` places what fits and runs the rest
on the CPU, so a Q8_0 reference (~27 GiB) works on the 24 GB XTX. `--process-output` includes
the output layer. The MTP block is not exercised by imatrix collection; its tensors quantize
without data (a warning, not an error, for IQ4_XS and up). Expect 20-40 min on a Q8_0
reference with 300k tokens; the Fit tab runs it as a job (it stops and restarts the instance).

## 11. Projector (mmproj)

llama.cpp cannot load a projector from inside the model file; `--mmproj` is always separate.
`llama-quantize` does quantize the CLIP projector: F16 885 MiB -> Q8_0 597 MiB (dry run).
With `MMPROJ_OFFLOAD=0` it lives in host RAM, not VRAM.

## 12. Next steps

| Phase | What | Needs |
|---|---|---|
| D | Build candidates (CPU, done: two built, every tensor as planned); verify them against the running model with the optimizer's models mode (suites, speed, depth curve, measured VRAM/GTT) | server stopped per model |
| KLD | Q8_0 reference logits, KL divergence per candidate | not started |
| imatrix | Q8_0 reference + llama-imatrix, then rebuild the winner with it | server stopped |
