# LexiPanel parameter reference

Generated from the panel's own parameter definitions (`PARAM_META` in `panel.py`, `sdcpp.meta()`, `audiocpp.meta()`), so it matches the tooltips in the UI word for word. Every setting is saved per instance in that instance's `params.env`; for the legacy `main` instance the launch script's defaults are overridden by `panel/params.env`.

Some descriptions mention measurements from the original host (a 7900 XTX box); treat those as that machine's numbers, not guarantees.

- [llama.cpp instances](#llamacpp-instances) — 164 settings
- [stable-diffusion.cpp instances](#stable-diffusioncpp-instances) — 31 settings
- [audio.cpp instances](#audiocpp-instances) — 13 settings, plus per-model options
- [Camelid instances](#camelid-instances) — 18 settings

## llama.cpp instances

One `llama-server` per instance. Settings the form does not expose can still be passed: the **All build options** list on the Parameters tab writes any flag the active build's `--help` accepts into `EXTRA_ARGS`.

| Group | Settings |
|---|---|
| [Backend](#llamacpp-backend) | 1 |
| [Offload](#llamacpp-offload) | 9 |
| [Context & memory](#llamacpp-context--memory) | 18 |
| [Long context (RoPE)](#llamacpp-long-context-rope) | 9 |
| [Throughput](#llamacpp-throughput) | 11 |
| [Speculative decoding](#llamacpp-speculative-decoding) | 9 |
| [Speculative (n-gram)](#llamacpp-speculative-n-gram) | 12 |
| [ROCm (HIP)](#llamacpp-rocm-hip) | 9 |
| [Vulkan (RADV)](#llamacpp-vulkan-radv) | 32 |
| [Vision](#llamacpp-vision) | 6 |
| [Generation](#llamacpp-generation) | 1 |
| [Reasoning](#llamacpp-reasoning) | 8 |
| [Sampling](#llamacpp-sampling) | 25 |
| [Safety](#llamacpp-safety) | 1 |
| [Server](#llamacpp-server) | 13 |

### llama.cpp Backend

#### `BACKEND` — Compute backend

**Type:** select · **Default:** `vulkan` · **Values:** `vulkan`, `rocm`, `cuda`, `cpu`

Which llama.cpp build serves the model. Each backend picks its own build in the Builds card - see there for what is actually installed, and never assume a version from this text. **vulkan** = RADV, the production path and the only one with measured numbers on this box. **rocm** = HIP/gfx1100, downloaded and script-ready but NOT yet benchmarked here. Switching needs a restart and reloads the full 15.7 GB model (~25 s). ROCm additionally needs its runtime installed: `sudo apt install -y libamdhip64-7 librocblas5 libhipblas3`. **cpu** = the plain ubuntu-x64 build, no GPU at all. On this 4-core E-2224G expect single-digit t/s on a 27B IQ4_XS - it is a fallback for when the GPU is unavailable, not a serving path. Each backend keeps its own build selection; see the Builds card.

### llama.cpp Offload

#### `OVERRIDE_TENSORS` — Tensor placement override

**Type:** text · **Default:** _(unset — engine default)_ · **Flag:** `--override-tensor`

`-ot/--override-tensor` _pattern=buffer_, comma-separated. Pins matching tensors to a buffer type, e.g. `exps=CPU` keeps MoE expert weights in host RAM while attention stays on the card - the standard way to run a MoE model larger than a 6 GB card. Host RAM is shared with every instance: check the RAM budget card.

#### `N_CPU_MOE` — MoE layers on CPU

**Type:** int · **Default:** _(unset — engine default)_ · **Flag:** `--n-cpu-moe` · **Values:** `4`, `8`, `16`, `32`

`-ncmoe/--n-cpu-moe N`. Keeps the expert weights of the first N layers on the CPU. Coarser than OVERRIDE_TENSORS, easier to tune: raise N until the model fits. Dense models ignore it.

#### `CPU_MOE` — All MoE experts on CPU

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--cpu-moe` · **Values:** `on`

`-cmoe/--cpu-moe`. Every expert on the CPU. The quickest way to get a large MoE model loading at all on a small card.

#### `SPEC_DRAFT_DEVICE` — Draft model device

**Type:** text · **Default:** _(unset — engine default)_ · **Flag:** `--spec-draft-device`

`--spec-draft-device`, e.g. `Vulkan1`. Runs the draft (or MTP sidecar) on another card, so its weights and KV stop competing with the target's experts for VRAM. Device names are the ones this instance's launch plan lists on the Status tab - with two drivers loaded the order is the loader's, not the PCI order.

#### `MMPROJ_DEVICE` — Vision projector device

**Type:** text · **Default:** _(unset — engine default)_ · **Flag:** `--mmproj-device`

`-mmdev/--mmproj-device`, e.g. `Vulkan1`, or `none` for CPU. Only takes effect with MMPROJ_OFFLOAD on.

#### `SPLIT_MODE` — Multi-GPU split mode

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--split-mode` · **Values:** `none`, `layer`, `row`, `tensor`

`-sm`. **layer** (stock) pipelines whole layers across the cards and is the one to use with two different cards. **row**/**tensor** split each weight and need fast links; the RTX 2060 sits in a chipset x4 slot.

#### `TENSOR_SPLIT` — Tensor split

**Type:** text · **Default:** _(unset — engine default)_ · **Flag:** `--tensor-split`

`-ts`, e.g. `24,6`: proportion of offloaded layers per device, in the instance's device order. Leave empty and place experts explicitly with OVERRIDE_TENSORS when most of the model lives on the CPU.

#### `MAIN_GPU` — Main GPU index

**Type:** int · **Default:** _(unset — engine default)_ · **Flag:** `--main-gpu`

`-mg`. Only meaningful with SPLIT_MODE none or row.

#### `EXTRA_ARGS` — Extra llama-server arguments

**Type:** text · **Default:** _(unset — engine default)_

Appended verbatim (shell-quoted) to the command line, for flags this panel does not model - e.g. `--split-mode none`. Applied by instance launches only; main's legacy scripts ignore it. The panel REFUSES draft-model flags here (`-md`, `--spec-draft-model` - trap 1) and anything the launch plan owns (port, host, model, log file, API key).

### llama.cpp Context & memory

#### `CTX` — Context size

**Type:** select · **Unit:** tokens · **Default:** `131072`

Total KV cache depth in tokens. **The single biggest VRAM lever.** KV scales linearly: on this model only 16 of 64 layers carry context-scaling KV (48 are Gated DeltaNet with a fixed recurrent state), which is why 244736 fits at all. Native max is 262144. Too high and the card runs near-full, which is what makes amdgpu evict weights into GTT - host RAM - and that is what OOM'd this box on 2026-09-04.

#### `KV_TYPE` — KV cache quant

**Type:** select · **Default:** `q8_0` · **Values:** `f32`, `f16`, `bf16`, `q8_0`, `q5_1`, `q5_0`, `q4_1`, `q4_0`, `iq4_nl`

Precision of the K and V caches. Cost is linear in bits: q8_0 is ~1.42x the bytes of q5_1, f16 is ~2.67x. Lower quant buys context depth and pays in long-chain consistency - the desktop notes record q4_0 'measurably costing output quality', which is why q8_0 was the compromise there. Applies to both K and V.

#### `CACHE_RAM` — Prompt cache (host RAM)

**Type:** select · **Unit:** MiB · **Default:** `4096` · **Values:** `0`, `-1`, `2048`, `4096`, `8192`, `12288`, `16384`, `20000`, `24576`, `32768`

**Host RAM**, not VRAM. Stores KV snapshots of previous prompts so a returning conversation restores instead of re-prefilling. At ~165k tokens a re-prefill costs minutes, so this is load-bearing for long agentic loops. Budget carefully: this box has 30.67 GB and the launch-script header sizes it at 8192 ('30G box, model mmap is ~16G'). It is only safe at 20000 if VRAM has headroom - if the GPU evicts to GTT, that eviction plus this cache is what exhausts RAM.

#### `CACHE_REUSE` — Cache reuse chunk

**Type:** int · **Unit:** tokens · **Default:** `256` · **Values:** `0`, `128`, `256`, `512`, `1024`

Minimum chunk size to salvage from the prompt cache via KV shifting when the prefix has drifted, instead of re-prefilling from scratch. 0 = off. 256 is the conventional value. Requires prompt caching (CACHE_RAM > 0). Aimed squarely at agentic loops where each turn appends to a long, slightly-shifted prompt.

#### `LOAD_MODE` — Model load mode

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--load-mode` · **Values:** `auto`, `none`, `mmap`, `mlock`, `mmap+mlock`, `dio`

`-lm/--load-mode`, **new in b10985**. This REPLACES the old `--no-mmap` and `--mlock`, which no longer exist - if you have notes telling you to pass those, they are stale. **auto** (stock) mmaps unless a device cannot. **mlock** pins the weights in RAM; on a 30 GB host already running a 15.7 GB model that is a fast route to the swap-thrash OOM of 2026-09-04, so treat it as a danger setting. **dio** uses DirectIO if available. Empty leaves the flag off.

#### `LAZY_MODE` — Lazy tensor reads

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--lazy-mode` · **Values:** `auto`, `on`, `off`

`-lzm/--lazy-mode`, new in b10985. Reads oversized tensors (e.g. per-layer embeddings) from disk on demand instead of keeping them resident; requires mmap. Stock is **auto** = on for tensors over 4 GiB. Nothing in this IQ4_XS file is that large, so this is a no-op here unless the model changes.

#### `KV_OFFLOAD` — KV cache on GPU

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--kv-offload / --no-kv-offload` · **Values:** `on`, `off`

`-kvo/--kv-offload` / `-nkvo/--no-kv-offload`. Stock is ENABLED - the KV cache lives in VRAM. Turning it off moves the whole cache to host RAM, which at ctx 245760 is many GB over PCIe every token. It is a diagnostic, not a tuning knob: use it to prove a hang is KV-related, not to buy headroom.

#### `SWA_FULL` — Full-size SWA cache

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--swa-full` · **Values:** `on`, `off`

`--swa-full`. Only meaningful for sliding-window-attention models; Qwen3.8 is not one, so this is inert here. Kept visible because a future model swap can make it suddenly matter - a wrong default on an SWA model silently costs either accuracy or a large multiple of KV memory.

#### `KV_UNIFIED` — Unified KV buffer

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--kv-unified / --no-kv-unified` · **Values:** `on`, `off`

`-kvu/--kv-unified` / `--no-kv-unified`. One KV buffer shared across all sequences instead of one per slot. Stock is enabled when the slot count is auto. This box runs a single slot (see Server slots), so it changes nothing until PARALLEL goes above 1.

#### `KV_UNIFIED_PER_SLOT` — KV per slot

**Type:** int · **Unit:** tokens · **Default:** _(unset — engine default)_ · **Flag:** `--kv-unified-per-slot` · **Values:** `32768`, `65536`, `131072`

`--kv-unified-per-slot`, new in b10985. Context limit per slot. If set **without** -c/--ctx-size the shared KV pool is sized to n_parallel*N. This panel always passes -c, so here it only caps each slot within the pool you already sized. Empty = unset = behaviour unchanged.

#### `CTX_CHECKPOINTS` — Context checkpoints

**Type:** int · **Default:** _(unset — engine default)_ · **Flag:** `--ctx-checkpoints` · **Values:** `0`, `4`, `8`, `16`, `32`

`-ctxcp/--ctx-checkpoints` (was --swa-checkpoints). Max context checkpoints kept per slot; stock is **32**. Checkpoints are what let the server rewind instead of re-prefilling, which at 245760 context is the difference between a resumed turn and a ~5 minute re-read. They are not free in memory, so this is the first thing to lower if the VRAM estimate is marginal and you would rather pay in prefill than in headroom.

#### `CHECKPOINT_MIN_STEP` — Checkpoint spacing

**Type:** int · **Unit:** tokens · **Default:** _(unset — engine default)_ · **Flag:** `--checkpoint-min-step` · **Values:** `0`, `4096`, `8192`, `16384`

`-cms/--checkpoint-min-step`. Minimum spacing between checkpoints; stock 8192, 0 = no minimum. Wider spacing = fewer checkpoints = less memory and coarser rewind granularity.

#### `CACHE_PROMPT` — Prompt caching

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--cache-prompt / --no-cache-prompt` · **Values:** `on`, `off`

`--cache-prompt` / `--no-cache-prompt`. Stock ENABLED, and it must stay enabled for CACHE_REUSE to do anything - `--cache-reuse` is documented as requiring it. Turning it off makes every request a cold prefill. Only reason to touch it is to measure what the cache is worth.

#### `CACHE_IDLE_SLOTS` — Cache idle slots

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--cache-idle-slots / --no-cache-idle-slots` · **Values:** `on`, `off`

`--cache-idle-slots` / `--no-cache-idle-slots`, new in b10985. Saves idle slots into the prompt cache when a new task arrives, and clears them when using unified KV. Stock enabled, and it requires cache-ram to be non-zero. Costs host RAM out of the CACHE_RAM budget, which the RAM calculator on this page already accounts for.

#### `CONTEXT_SHIFT` — Context shift

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--context-shift / --no-context-shift` · **Values:** `on`, `off`

`--context-shift` / `--no-context-shift`. **Upstream changed the default to DISABLED**, which is why long sessions now stop at the context limit instead of silently sliding the window and dropping the head of the conversation. Turning it on trades a hard stop for silent amnesia. With PRESERVE_THINKING on, what gets dropped first is the oldest reasoning, so the model loses its own earlier conclusions without saying so.

#### `FIT` — Auto-fit to VRAM

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--fit` · **Values:** `on`, `off`

`-fit/--fit`, **new and stock-ENABLED in b10985**. llama.cpp will quietly adjust arguments you did _not_ set so the model fits device memory. This panel sets ctx, ngl, batch and ubatch explicitly, so there is little left for it to move - but it is the reason a config can now load where the same numbers would have failed on b10766. If you are bisecting a memory regression against the old build, set this to **off** first so the two builds are actually comparable.

#### `FIT_TARGET` — Auto-fit margin

**Type:** int · **Unit:** MiB · **Default:** _(unset — engine default)_ · **Flag:** `--fit-target` · **Values:** `512`, `1024`, `2048`

`-fitt/--fit-target`. Margin per device left free by --fit; stock 1024 MiB. This box runs with roughly that much headroom in total, so the margin and the working config are the same size - raise it only together with a lower CTX.

#### `FIT_CTX` — Auto-fit min context

**Type:** int · **Unit:** tokens · **Default:** _(unset — engine default)_ · **Flag:** `--fit-ctx` · **Values:** `4096`, `32768`, `131072`

`-fitc/--fit-ctx`. Floor on the context --fit is allowed to shrink to; stock 4096. Inert while CTX is set explicitly.

### llama.cpp Long context (RoPE)

#### `ROPE_SCALING` — RoPE scaling

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--rope-scaling` · **Values:** `none`, `linear`, `yarn`

`--rope-scaling {none,linear,yarn}`. The context dropdown offers up to 524288 while this model's native training context is **262144**. Anything above that is extrapolation and needs RoPE scaling to be coherent rather than merely allocated - the server will happily allocate a 512k cache and produce degrading output past the native limit with no warning. yarn is the usual choice for Qwen. Empty = model default.

#### `ROPE_SCALE` — RoPE scale

**Type:** text · **Default:** _(unset — engine default)_ · **Flag:** `--rope-scale`

`--rope-scale N`. Context expansion factor. With linear scaling, 2 doubles the usable context. Empty = model default.

#### `ROPE_FREQ_BASE` — RoPE freq base

**Type:** text · **Default:** _(unset — engine default)_ · **Flag:** `--rope-freq-base`

`--rope-freq-base N`. NTK-aware base frequency. Empty = loaded from the GGUF, which is almost always what you want.

#### `ROPE_FREQ_SCALE` — RoPE freq scale

**Type:** text · **Default:** _(unset — engine default)_ · **Flag:** `--rope-freq-scale`

`--rope-freq-scale N`. Inverse form of --rope-scale (expands by 1/N). Set one or the other, never both.

#### `YARN_ORIG_CTX` — YaRN original ctx

**Type:** int · **Unit:** tokens · **Default:** _(unset — engine default)_ · **Flag:** `--yarn-orig-ctx` · **Values:** `32768`, `131072`, `262144`

`--yarn-orig-ctx`. The model's _training_ context, which YaRN scales up from. 0 = read from the model. For this model that is 262144.

#### `YARN_EXT_FACTOR` — YaRN ext factor

**Type:** text · **Default:** _(unset — engine default)_ · **Flag:** `--yarn-ext-factor`

`--yarn-ext-factor`. Extrapolation mix; -1 = model default, 0.0 = full interpolation.

#### `YARN_ATTN_FACTOR` — YaRN attn factor

**Type:** text · **Default:** _(unset — engine default)_ · **Flag:** `--yarn-attn-factor`

`--yarn-attn-factor`. Attention magnitude scaling. -1 = default.

#### `YARN_BETA_FAST` — YaRN beta fast

**Type:** text · **Default:** _(unset — engine default)_ · **Flag:** `--yarn-beta-fast`

`--yarn-beta-fast`. Low correction dim (beta). -1 = default.

#### `YARN_BETA_SLOW` — YaRN beta slow

**Type:** text · **Default:** _(unset — engine default)_ · **Flag:** `--yarn-beta-slow`

`--yarn-beta-slow`. High correction dim (alpha). -1 = default.

### llama.cpp Throughput

#### `BATCH` — Logical batch (-b)

**Type:** select · **Unit:** tokens · **Default:** `512` · **Values:** `128`, `256`, `512`, `1024`, `2048`, `4096`, `8192`

How many prompt tokens are submitted per prefill step. Higher = faster prefill, little VRAM cost (the physical buffer is set by UBATCH, not this). b10766's own default is 2048; this box ran 512 for a while, which was throttling prefill for no memory saving.

#### `UBATCH` — Physical batch (-ub)

**Type:** select · **Unit:** tokens · **Default:** `512` · **Values:** `64`, `128`, `256`, `512`, `1024`, `2048`

Physical micro-batch actually evaluated at once. **This is the one that sizes the prefill compute buffer**, so it is the VRAM knob and the transient-spike knob. The desktop script's ITEM 6 device-lost crash on deep-context prefill lists `-ub 256` as the first mitigation, explicitly not lowering -b. Leave at 512 unless chasing a device-lost.

#### `NGL` — GPU layers

**Type:** select · **Default:** `99` · **Values:** `0`, `16`, `32`, `48`, `64`, `80`, `99`

Layers offloaded to the GPU. 99 = all of them. Anything less puts transformer layers on a 4-core Xeon E-2224G and collapses throughput. Note that pinning this makes llama.cpp's auto-fitter abort ('n_gpu_layers already set by user'), so nothing caps an over-budget config for you - size memory by hand.

#### `THREADS` — CPU threads

**Type:** select · **Default:** `4` · **Values:** `1`, `2`, `3`, `4`

CPU threads for the non-offloaded path and sampling. This host is a Xeon E-2224G: 4 cores, 4 threads, no SMT. Above 4 oversubscribes and hurts. Used for both --threads and --threads-batch.

#### `FLASH_ATTN` — Flash attention

**Type:** select · **Default:** `on` · **Flag:** `--flash-attn` · **Values:** `auto`, `on`, `off`

`-fa/--flash-attn`. Was **hardcoded to 'on'** in all three launch scripts until 2026-09-15 and is now a real setting. It is what makes the quantised KV types usable: without it the q5_1/q4_1 cache paths fall back and the memory estimate on this page stops being true. Stock upstream is 'auto'; this box ships 'on' because that is the configuration every measurement here was taken under.

#### `PARALLEL` — Server slots

**Type:** select · **Default:** `1` · **Flag:** `--parallel` · **Values:** `1`, `2`, `4`, `8`

`-np/--parallel`. Was **hardcoded to 1** until 2026-09-15. Each slot gets its own share of the KV pool, so going to 2 slots at a fixed CTX halves the depth each conversation can reach. Upstream's stock is -1 (auto). Keep it at 1 on this box: 24 GB with ~1.4 GB spare does not have room for a second full-depth conversation, and the VRAM calculator above assumes one.

#### `THREADS_HTTP` — HTTP threads

**Type:** int · **Default:** _(unset — engine default)_ · **Flag:** `--threads-http` · **Values:** `-1`, `1`, `2`, `4`

`--threads-http`. Threads serving HTTP, separate from inference threads. Stock -1 = auto. On a 4-core E-2224G every HTTP thread competes with the 4 inference threads, so raising it costs decode throughput.

#### `BACKEND_SAMPLING` — Backend sampling

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--backend-sampling` · **Values:** `on`, `off`

`-bs/--backend-sampling`, new in b10985 and marked **experimental** upstream. Moves the sampler onto the GPU. Untested on RADV here, and this box's failure mode for a bad GPU path is a compute-ring timeout or a hard lock, not an error message. Leave empty unless you are deliberately testing it with a short context.

#### `OP_OFFLOAD` — Offload host ops

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--op-offload / --no-op-offload` · **Values:** `on`, `off`

`--op-offload` / `--no-op-offload`. Whether host tensor operations are pushed to the device; stock enabled. Disabling moves work back to the CPU - a diagnostic for suspected backend op bugs, not a speed knob.

#### `REPACK` — Weight repacking

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--repack / --no-repack` · **Values:** `on`, `off`

`--repack` / `-nr/--no-repack`. Repacks weights into a layout the CPU kernels prefer; stock enabled. Matters for BACKEND=cpu, irrelevant at NGL 99 where nothing runs on the CPU.

#### `NO_HOST` — Bypass host buffer

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--no-host` · **Values:** `on`, `off`

`--no-host`, new in b10985. Bypasses the host buffer so extra buffer types can be used. Interacts directly with the GTT-placement trap: it changes where staging buffers live. If you set it, re-check VRAM vs GTT (healthy is ~23,000 MiB VRAM against under ~1,000 MiB GTT) before trusting any timing.

### llama.cpp Speculative decoding

#### `SPEC_TYPE` — Spec type

**Type:** select · **Default:** `draft-mtp` · **Values:** `none`, `draft-mtp`, `draft-simple`, `draft-eagle3`, `draft-dflash`, `draft-dspark`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`, `draft-dflash,ngram-mod`, `draft-mtp,ngram-mod`, `draft-simple,ngram-mod`

**draft-mtp** uses the model's embedded NextN/MTP head - measured 0.82 draft acceptance, mean accepted length 2.64, for only +370 MiB. Worth roughly 1.8x on decode. **none** disables speculation. There is no reason to turn this off on this model.

#### `SPEC_N_MAX` — Draft depth

**Type:** select · **Default:** `2` · **Values:** `1`, `2`, `3`, `4`, `5`

How many tokens the draft head proposes per step. At the measured 0.82 acceptance and mean length 2.64 the depth-2 ceiling is hit nearly every draft, which is the condition under which depth 3 starts paying. The n=2/3/4 sweep is still an open item from the desktop script.

#### `SPEC_DRAFT_MODEL` — Draft model file

**Type:** select · **Default:** _(unset — engine default)_

**Leave empty.** Empty means the embedded MTP head. Pointing this at the full-size target loads a SECOND complete 15.7 GB copy plus a second KV cache and has hard-locked this host twice (2026-09-02). Only a small sidecar is ever valid. Each option below shows its own VRAM cost.

**On FastMTP:** the 862 MB sidecar cannot be used with the prebuilt binaries. It carries a _trimmed 32k draft vocabulary_ plus a `d2t` remap tensor, and stock llama.cpp hard-asserts the full 248320 vocab - hence `expected 5120, 248320, got 5120, 32768`. It needs a source build with HauhauCS-FastMTP-llama.cpp.patch. Merging it into the target GGUF does NOT help: the blocker is the loader, not the file layout. Panel and launch script both reject anything over 4 GB.

#### `SPEC_N_MIN` — Draft n-min

**Type:** int · **Default:** _(unset — engine default)_ · **Flag:** `--spec-draft-n-min` · **Values:** `0`, `1`, `2`

`--spec-draft-n-min`. Floor on drafted tokens per step; stock 0. Raising it forces the draft head to commit even when its confidence is low, which on this box shows up as a falling acceptance rate (measured 0.56-0.82, content-dependent) rather than as an error.

#### `SPEC_P_MIN` — Draft p-min

**Type:** float · **Default:** `0` · **Values:** `0`, `0.1`, `0.5`, `0.9`

`--spec-draft-p-min`. Minimum probability before a drafted token is proposed; stock 0.00 (greedy). The launch scripts have passed 0 explicitly since before the panel existed and it is now a real setting rather than a literal.

#### `SPEC_P_SPLIT` — Draft p-split

**Type:** float · **Default:** _(unset — engine default)_ · **Flag:** `--spec-draft-p-split` · **Values:** `0.1`, `0.5`

`--spec-draft-p-split`. Split probability for the draft tree; stock 0.10. Only used by tree-style draft types.

#### `SPEC_DRAFT_KV_TYPE` — Draft KV type

**Type:** select · **Default:** _(unset — engine default)_ · **Values:** `f32`, `f16`, `bf16`, `q8_0`, `q5_1`, `q5_0`, `q4_1`, `q4_0`, `iq4_nl`

`-ctkd/-ctvd` (`--spec-draft-type-k/-v`), new names in b10985. KV cache type for the DRAFT context, set independently of the target's KV_TYPE; stock f16. With `draft-mtp` the draft context is the ~370 MiB one created against the target, so this is a small saving - but it is a real one when headroom is measured in hundreds of MiB. Empty = f16.

#### `SPEC_DRAFT_NGL` — Draft GPU layers

**Type:** select · **Default:** _(unset — engine default)_ · **Values:** `all`, `auto`, `0`

`-ngld/--spec-draft-ngl`. Accepts a number, **auto** or **all**. The launch scripts pass `all` automatically whenever a real sidecar is set in SPEC_DRAFT_MODEL; this overrides that. Irrelevant for the embedded MTP head, which has no separate layers to place.

#### `SPEC_DRAFT_BACKEND_SAMPLING` — Draft backend sampling

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--spec-draft-backend-sampling / --no-spec-draft-backend-sampling` · **Values:** `on`, `off`

`--spec-draft-backend-sampling` / `--no-...`, new in b10985 and stock **enabled**. Offloads draft sampling to the backend. Unlike the target-side `-bs` this one is on by default, so if you are chasing a speculative-decoding fault on RADV, turning this OFF is the cheap first bisect.

### llama.cpp Speculative (n-gram)

#### `SPEC_NGRAM_MOD_N_MIN` — ngram-mod n-min

**Type:** int · **Default:** _(unset — engine default)_ · **Flag:** `--spec-ngram-mod-n-min` · **Values:** `48`

`--spec-ngram-mod-n-min`, stock 48. Only read when SPEC_TYPE is **ngram-mod**.

#### `SPEC_NGRAM_MOD_N_MAX` — ngram-mod n-max

**Type:** int · **Default:** _(unset — engine default)_ · **Flag:** `--spec-ngram-mod-n-max` · **Values:** `64`

`--spec-ngram-mod-n-max`, stock 64. Only read when SPEC_TYPE is **ngram-mod**. This is the flag that replaced the removed --draft-max for ngram types.

#### `SPEC_NGRAM_MOD_N_MATCH` — ngram-mod match len

**Type:** int · **Default:** _(unset — engine default)_ · **Flag:** `--spec-ngram-mod-n-match` · **Values:** `24`

`--spec-ngram-mod-n-match`, stock 24. Lookup length for ngram-mod. Replaced the removed --spec-ngram-size-n.

#### `SPEC_NGRAM_SIMPLE_SIZE_N` — ngram-simple lookup N

**Type:** int · **Default:** _(unset — engine default)_ · **Flag:** `--spec-ngram-simple-size-n` · **Values:** `12`

`--spec-ngram-simple-size-n`, stock 12. Only read when SPEC_TYPE is **ngram-simple**.

#### `SPEC_NGRAM_SIMPLE_SIZE_M` — ngram-simple draft M

**Type:** int · **Default:** _(unset — engine default)_ · **Flag:** `--spec-ngram-simple-size-m` · **Values:** `48`

`--spec-ngram-simple-size-m`, stock 48.

#### `SPEC_NGRAM_SIMPLE_MIN_HITS` — ngram-simple min hits

**Type:** int · **Default:** _(unset — engine default)_ · **Flag:** `--spec-ngram-simple-min-hits` · **Values:** `1`

`--spec-ngram-simple-min-hits`, stock 1.

#### `SPEC_NGRAM_MAP_K_SIZE_N` — ngram-map-k lookup N

**Type:** int · **Default:** _(unset — engine default)_ · **Flag:** `--spec-ngram-map-k-size-n` · **Values:** `12`

`--spec-ngram-map-k-size-n`, stock 12. Only read when SPEC_TYPE is **ngram-map-k**.

#### `SPEC_NGRAM_MAP_K_SIZE_M` — ngram-map-k draft M

**Type:** int · **Default:** _(unset — engine default)_ · **Flag:** `--spec-ngram-map-k-size-m` · **Values:** `48`

`--spec-ngram-map-k-size-m`, stock 48.

#### `SPEC_NGRAM_MAP_K_MIN_HITS` — ngram-map-k min hits

**Type:** int · **Default:** _(unset — engine default)_ · **Flag:** `--spec-ngram-map-k-min-hits` · **Values:** `1`

`--spec-ngram-map-k-min-hits`, stock 1.

#### `SPEC_NGRAM_MAP_K4V_SIZE_N` — ngram-map-k4v lookup N

**Type:** int · **Default:** _(unset — engine default)_ · **Flag:** `--spec-ngram-map-k4v-size-n` · **Values:** `12`

`--spec-ngram-map-k4v-size-n`, stock 12. Only read when SPEC_TYPE is **ngram-map-k4v**.

#### `SPEC_NGRAM_MAP_K4V_SIZE_M` — ngram-map-k4v draft M

**Type:** int · **Default:** _(unset — engine default)_ · **Flag:** `--spec-ngram-map-k4v-size-m` · **Values:** `48`

`--spec-ngram-map-k4v-size-m`, stock 48.

#### `SPEC_NGRAM_MAP_K4V_MIN_HITS` — ngram-map-k4v min hits

**Type:** int · **Default:** _(unset — engine default)_ · **Flag:** `--spec-ngram-map-k4v-min-hits` · **Values:** `1`

`--spec-ngram-map-k4v-min-hits`, stock 1.

### llama.cpp ROCm (HIP)

#### `GGML_CUDA_REGISTER_HOST` — Register host memory

**Type:** text · **Default:** _(unset — engine default)_

**Prime suspect for the 2026-09-04 ROCm prefill stall.** Controls whether ggml registers host buffers with the GPU (hipHostRegister). That registration is what the kernel was thrashing on when prefill halved batch-over-batch (27.8 → 14.1 t/s) while logging `amdgpu_amdkfd_restore_userptr_worker hogged CPU`. Set **0** to disable and see whether prefill recovers. Vulkan equivalent: none - RADV does not register host memory this way. Empty = unset = stock.

#### `GGML_CUDA_ENABLE_UNIFIED_MEMORY` — Unified memory

**Type:** text · **Default:** _(unset — engine default)_

**The HIP analogue of the Vulkan sysmem-fallback trap.** Set it and HIP will silently back device allocations with host memory over PCIe - the same ~20x slowdown that cost this box 2.79 t/s prefill under Vulkan, with no error logged. It is opt-in, so the correct value is **empty (unset)**. Translation: `GGML_VK_ALLOW_SYSMEM_FALLBACK=0` under Vulkan is a guard you must SET; this is a footgun you must NOT set. Opposite polarity - do not 'convert' one to the other.

#### `GGML_CUDA_NO_PINNED` — Disable pinned host memory

**Type:** text · **Default:** _(unset — engine default)_

Set to 1 to stop using page-locked host buffers for transfers. Pinned memory makes host↔device copies faster but pins pages, which interacts with the same KFD userptr machinery implicated in the prefill stall. Worth trying alongside REGISTER_HOST=0. Vulkan equivalent: none. Empty = unset = stock.

#### `GGML_CUDA_DISABLE_GRAPHS` — Disable HIP graphs

**Type:** text · **Default:** _(unset — engine default)_

Set to 1 to stop capturing the decode step as a replayable HIP graph. Graphs cut per-token launch overhead, which matters most at high decode rates; disabling is a diagnostic for graph-capture bugs, not a tuning win. The server log reports `graphs reused = N` so you can see whether they are being hit. Vulkan equivalent: none. Empty = unset = stock.

#### `GGML_CUDA_GRAPH_OPT` — HIP graph optimisation

**Type:** text · **Default:** _(unset — engine default)_

Tunes graph-capture optimisation. Leave empty unless chasing a specific graph problem; DISABLE_GRAPHS is the blunter and better-understood diagnostic. Empty = unset = stock.

#### `GGML_CUDA_DISABLE_FUSION` — Disable kernel fusion

**Type:** text · **Default:** _(unset — engine default)_

Set to 1 to stop fusing adjacent ops into single kernels. Fusion is normally a win; disabling it isolates whether a fused kernel is miscompiled or slow on gfx1100. Diagnostic, not tuning. Empty = unset = stock.

#### `GGML_CUDA_DEVICES` — Device selection

**Type:** text · **Default:** _(unset — engine default)_

Which HIP devices ggml may use, e.g. `0`. Translation: this is the HIP counterpart of `GGML_VK_VISIBLE_DEVICES`. Less critical here than under Vulkan - the Intel UHD P630 cannot enumerate under HSA at all, so there is no wrong-GPU trap to guard against. The launch script also sets HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES, which are read by the ROCm runtime rather than by ggml. Empty = unset = stock.

#### `HSA_ENABLE_SDMA` — SDMA copy engines

**Type:** text · **Default:** _(unset — engine default)_

Read by the ROCr runtime, not by ggml. Set to 0 to route host↔device copies through compute kernels instead of the dedicated SDMA engines. SDMA problems show up as stalled or extremely slow transfers, which is the shape of the prefill stall - so 0 is a reasonable third thing to try. Vulkan equivalent: none. Empty = unset = stock.

#### `GGML_CUDA_CUBLAS_COMPUTE_TYPE` — cuBLAS compute type

**Type:** text · **Default:** _(unset — engine default)_

Overrides the compute type hipBLAS uses for matmul, e.g. forcing FP32 accumulation where the stock path would accumulate in FP16. The usual reason to set it is numerically wrong output under ROCm that is fine under Vulkan. Ignored entirely when BACKEND=vulkan.

### llama.cpp Vulkan (RADV)

#### `GGML_VK_DISABLE_COOPMAT` — Disable coopmat

**Type:** select · **Default:** _(unset — engine default)_ · **Values:** `0`, `1`

Cooperative-matrix (matrix core) kernels. `--list-devices` reports this card as **matrix cores: KHR_coopmat**, so they are in use. Set 1 to fall back to plain shaders - a large expected slowdown, useful only to test whether a coopmat kernel is miscompiled on RADV NAVI31. ROCm equivalent: none.

#### `GGML_VK_DISABLE_COOPMAT2` — Disable coopmat2

**Type:** select · **Default:** _(unset — engine default)_ · **Values:** `0`, `1`

Disables the NV_cooperative_matrix2 path specifically. This card advertises KHR_coopmat rather than coopmat2, so setting this should change nothing here - it is listed for completeness and for diffing against other hardware.

#### `GGML_VK_DISABLE_F16` — Disable fp16

**Type:** select · **Default:** _(unset — engine default)_ · **Values:** `0`, `1`

Forces fp32 maths. `--list-devices` reports **fp16: 1** on this card, so fp16 is active and disabling it costs both speed and VRAM. Diagnostic only.

#### `GGML_VK_DISABLE_BFLOAT16` — Disable bf16

**Type:** select · **Default:** _(unset — engine default)_ · **Values:** `0`, `1`

This card reports **bf16: 0** - bf16 is already unavailable, so this is a no-op here. Relevant only on hardware that has it.

#### `GGML_VK_DISABLE_FUSION` — Disable kernel fusion

**Type:** select · **Default:** _(unset — engine default)_ · **Values:** `0`, `1`

Stops fusing adjacent ops into one shader. Fusion is normally a win; disabling isolates a miscompiled fused kernel. Direct counterpart of ROCm's `GGML_CUDA_DISABLE_FUSION`.

#### `GGML_VK_DISABLE_GRAPH_OPTIMIZE` — Disable graph optimise

**Type:** select · **Default:** _(unset — engine default)_ · **Values:** `0`, `1`

Turns off compute-graph reordering. Rough counterpart of ROCm's `GGML_CUDA_GRAPH_OPT`. Diagnostic.

#### `GGML_VK_DISABLE_MMVQ` — Disable MMVQ

**Type:** select · **Default:** _(unset — engine default)_ · **Values:** `0`, `1`

Quantised matrix-vector kernels - the hot path for **decode**, where one token at a time multiplies against quantised weights. Disabling will hurt decode noticeably. Pair with FORCE_MMVQ to A/B the same path.

#### `GGML_VK_FORCE_MMVQ` — Force MMVQ

**Type:** select · **Default:** _(unset — engine default)_ · **Values:** `0`, `1`

Forces the quantised matrix-vector path even where heuristics would pick a matrix kernel. Occasionally a win at batch 1; measure, do not assume.

#### `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT` — Disable int dot

**Type:** select · **Default:** _(unset — engine default)_ · **Values:** `0`, `1`

Integer dot-product acceleration for quantised weights. This card reports **int dot: 1**, so it is in use and matters for an IQ4_XS model. Diagnostic.

#### `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM` — Disable host-visible VRAM

**Type:** select · **Default:** _(unset — engine default)_ · **Values:** `0`, `1`

**Directly relevant to this box.** Resizable BAR makes all 24560 MiB of VRAM host-visible (confirmed: vis_vram_total == vram_total). This switch makes ggml stop using that window. If ReBAR ever regresses in BIOS, behaviour here is worth comparing. Do not set it casually - host-visible VRAM is what makes uploads fast.

#### `GGML_VK_PREFER_HOST_MEMORY` — Prefer host memory

**Type:** select · **Default:** _(unset — engine default)_ · **Values:** `0`, `1`

**Do not set this on this box.** It biases allocation towards host RAM, which is the failure mode that cost 2.79 t/s prefill here. It is the opposite of what you want; `GGML_VK_ALLOW_SYSMEM_FALLBACK=0` exists to prevent exactly this.

#### `GGML_VK_ENABLE_MEMORY_PRIORITY` — Memory priority

**Type:** select · **Default:** _(unset — engine default)_ · **Values:** `0`, `1`

Requests VK_EXT_memory_priority so the driver is told these allocations matter. Plausibly useful against the runtime-suspend eviction seen on this host - it is a hint to the driver, not a guarantee. Untested here; worth an experiment.

#### `GGML_VK_FORCE_MAX_ALLOCATION_SIZE` — Max allocation size

**Type:** text · **Default:** _(unset — engine default)_

Caps a single Vulkan allocation, in bytes. Some drivers fail on very large single buffers; splitting can work around that. Empty = driver limit.

#### `GGML_VK_SUBALLOCATION_BLOCK_SIZE` — Suballocation block

**Type:** text · **Default:** _(unset — engine default)_

Block size ggml suballocates within, in bytes. Affects fragmentation on a card already running near full. Empty = default.

#### `GGML_VK_MAX_NODES_PER_SUBMIT` — Max nodes per submit

**Type:** text · **Default:** _(unset — engine default)_

How many graph nodes go into one command-buffer submission. Larger = less submission overhead, but a longer single GPU job - and an over-long submission is what trips `ring ... timeout`, the desktop script's ITEM 6 device-lost. Lower it if you see ring timeouts. Empty = default.

#### `GGML_VK_DISABLE_ASYNC` — Disable async

**Type:** select · **Default:** _(unset — engine default)_ · **Values:** `0`, `1`

Serialises submissions instead of overlapping them. Costs throughput; useful to make a device-lost or ordering bug reproducible.

#### `GGML_VK_MEMORY_LOGGER` — Memory logger

**Type:** select · **Default:** _(unset — engine default)_ · **Values:** `0`, `1`

**Diagnostic.** Logs every allocation and whether it landed on device or host - the definitive answer to 'is the model actually on the card'. Verbose; turn on to investigate placement, then off.

#### `GGML_VK_PERF_LOGGER` — Perf logger

**Type:** select · **Default:** _(unset — engine default)_ · **Values:** `0`, `1`

**Diagnostic.** Per-op GPU timings, which is how you find which kernel is slow rather than guessing from end-to-end t/s. Adds overhead - measure with it off.

#### `GGML_VK_VISIBLE_DEVICES` — Visible devices

**Type:** text · **Default:** _(unset — engine default)_

Restricts which Vulkan devices ggml will enumerate, by index. The launch script already pins this to **0** together with VK_DRIVER_FILES, which is what keeps the Intel UHD P630 iGPU out of the picture (trap 4). Setting it here overrides that pin, and getting it wrong means the model is served by the iGPU or fails to find a device at all.

#### `GGML_VK_ALLOW_GRAPHICS_QUEUE` — Allow graphics queue

**Type:** text · **Default:** _(unset — engine default)_

Permits compute work on the graphics queue when no dedicated compute queue is usable. Relevant to the ring timeouts recorded on 2026-09-03 (`ring comp_X.Y.Z timeout`): those were compute-ring hangs, and this changes which ring the work lands on. That makes it an experiment worth running against that specific failure, not a speed setting.

#### `GGML_VK_ASYNC_USE_TRANSFER_QUEUE` — Async transfer queue

**Type:** text · **Default:** _(unset — engine default)_

Uses the dedicated transfer queue for async copies. Pairs with GGML_VK_DISABLE_ASYNC - disable async entirely, or keep it and move the copies to their own queue.

#### `GGML_VK_SERIALIZE_SUBMISSIONS` — Serialize submissions

**Type:** text · **Default:** _(unset — engine default)_

Forces command-buffer submissions to be serialised. Slow by design; it is a debugging aid for exactly the class of fault this box hits, where a GPU hang has no error attached to it. Set it to 1 to find out whether concurrency is what is wedging the ring.

#### `GGML_VK_SYNC_LOGGER` — Sync logger

**Type:** text · **Default:** _(unset — engine default)_

Logs synchronisation events. Very high volume; the engine log lives on tmpfs, so leaving this on will consume host RAM and the RAM_FLOOR_MB watchdog will eventually kill the server over it.

#### `GGML_VK_PIPELINE_STATS` — Pipeline statistics

**Type:** text · **Default:** _(unset — engine default)_

Dumps per-pipeline statistics at shutdown. Diagnostic only, no runtime cost worth worrying about.

#### `GGML_VK_DEBUG_MARKERS` — Debug markers

**Type:** text · **Default:** _(unset — engine default)_

Emits Vulkan debug markers for capture tools (RenderDoc, RGP). Useless without a capture tool attached, and this host is headless.

#### `GGML_VK_PERF_LOGGER_CONCURRENT` — Perf logger concurrent

**Type:** text · **Default:** _(unset — engine default)_

Concurrent variant of GGML_VK_PERF_LOGGER. Only read when the perf logger is on.

#### `GGML_VK_PERF_LOGGER_FREQUENCY` — Perf logger frequency

**Type:** text · **Default:** _(unset — engine default)_

Sampling frequency for the perf logger. Only read when the perf logger is on.

#### `GGML_VK_DISABLE_MULTI_ADD` — Disable multi-add

**Type:** text · **Default:** _(unset — engine default)_

Disables the fused multi-add path. One of the fusion knobs to bisect when output is subtly wrong rather than absent - a wrong fused kernel produces plausible garbage, not a crash.

#### `GGML_VK_DISABLE_DOT2` — Disable dot2

**Type:** text · **Default:** _(unset — engine default)_

Disables the 2-wide dot-product path. Companion to GGML_VK_DISABLE_INTEGER_DOT_PRODUCT, which the panel already exposes.

#### `GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR` — Disable coopmat2 decode vector

**Type:** text · **Default:** _(unset — engine default)_

Disables the coopmat2 vector path used during decode specifically, leaving it enabled for prefill. Finer-grained than GGML_VK_DISABLE_COOPMAT2, so it can separate a decode-only fault from a general coopmat2 problem - useful here, where prefill (~690 t/s) has always been healthy and decode is where the faults land.

#### `GGML_VK_FA_SPARSE_DISABLE` — Disable sparse flash-attn

**Type:** text · **Default:** _(unset — engine default)_

Disables the sparse flash-attention path. Reach for this before turning FLASH_ATTN off entirely: it keeps the quantised KV types working while removing only the sparse kernel.

#### `GGML_VK_FORCE_MAX_BUFFER_SIZE` — Force max buffer size

**Type:** text · **Default:** _(unset — engine default)_

Caps any single Vulkan buffer, in bytes. Distinct from GGML_VK_FORCE_MAX_ALLOCATION_SIZE, which the panel already exposes: allocation size bounds a device allocation, this bounds a buffer object within one. Both matter on RADV, where one oversized buffer fails the whole load.

### llama.cpp Vision

#### `MMPROJ` — Projector file

**Type:** select · **Default:** `/home/admin/models/mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf`

The vision projector, which must come from the **same repo as the model in MODEL**. Hardcoded in the launch script until 2026-09-15, so switching model used to keep the old projector. A mismatch does **not** error: these Qwen3.8 projectors share a shape (qwen3vl_merger, 5120 projection dim, 334 tensors), so the wrong one loads cleanly and only the image answers are wrong. Options that do not match MODEL are marked [other build].

#### `USE_MMPROJ` — Enable vision

**Type:** bool · **Default:** `1`

Loads the BF16 multimodal projector so the server accepts images. 1 = on. Turning it off entirely frees the most VRAM but loses image input.

#### `MMPROJ_OFFLOAD` — Projector on GPU

**Type:** bool · **Default:** `0`

1 = projector on-card (stock). 0 = `--no-mmproj-offload`, projector on the CPU. Off-card returns roughly 1.7 GB of VRAM - ~848 MiB for the projector plus the ~885 MiB mtmd worst-case buffer the desktop measured - and removes that buffer as a transient spike. Spikes are what trigger the GTT eviction. Vision still works; image encoding is slower. Cheap trade if images are occasional and text is constant.

#### `IMAGE_MIN_TOKENS` — Image min tokens

**Type:** int · **Default:** `1024` · **Values:** `256`, `512`, `1024`, `2048`

`--image-min-tokens`. Floor on tokens spent per image by a dynamic-resolution vision model. Was **hardcoded to 1024** in all three launch scripts until 2026-09-15. Every image now costs at least this much context, so at 1024 a handful of screenshots is a meaningful bite out of even a 245k window. Empty = read from the model.

#### `IMAGE_MAX_TOKENS` — Image max tokens

**Type:** int · **Default:** _(unset — engine default)_ · **Flag:** `--image-max-tokens` · **Values:** `1024`, `2048`, `4096`

`--image-max-tokens`. Ceiling per image. This is the one to set if a large screenshot is blowing out the context - it caps the cost instead of letting resolution decide it. Empty = read from the model.

#### `MTMD_BATCH_MAX_TOKENS` — Image encode batch

**Type:** int · **Default:** _(unset — engine default)_ · **Flag:** `--mtmd-batch-max-tokens` · **Values:** `512`, `1024`, `2048`

`--mtmd-batch-max-tokens`, stock 1024. Image tokens per encode batch. This sets the size of a transient VRAM spike during encoding, and transient spikes are what push amdgpu into evicting weights to GTT on this card. Lower it before lowering CTX if vision is what destabilises a config.

### llama.cpp Generation

#### `N_PREDICT` — Max tokens per reply

**Type:** int · **Default:** `16000` · **Values:** `1024`, `2048`, `4096`, `8192`, `16000`, `32000`

Server-side ceiling on generated tokens for one request. A client asking for more is capped here.

### llama.cpp Reasoning

#### `REASONING_EFFORT` — Reasoning effort

**Type:** select · **Default:** `medium` · **Values:** `default`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`

A **chat-template kwarg, not a server flag** - it injects an instruction string into every prompt. This template aliases **high to xhigh**, its maximum; there is no separate middle-high rung. xhigh injects a 333-character think-harder instruction on every prompt, medium injects nothing, low instructs brevity. The desktop log records xhigh + preserved thinking as the exact combination that exhausted a 150-iteration agentic run after five compactions. medium is the right setting for coding loops.

#### `REASONING_BUDGET` — Reasoning budget

**Type:** int · **Default:** `-1` · **Values:** `-1`, `1024`, `2048`, `4096`, `8192`, `16384`

Hard cap on thinking tokens. -1 = unrestricted.

#### `PRESERVE_THINKING` — Preserve thinking

**Type:** bool · **Default:** `true`

Keeps each turn's chain-of-thought in the context of every later turn. **Quadratic context growth from reasoning alone** - the desktop log blames this for forcing five compactions in one run, and each compaction rewrites the prefix and destroys the prefix-cache reuse this server otherwise gets at 0.94-0.99 similarity. Improves multi-turn coherence; expensive on long loops.

#### `MAX_TOOL_RESPONSE_CHARS` — Max tool response chars

**Type:** int · **Default:** `8000` · **Values:** `2000`, `4000`, `8000`, `9000`, `16000`

Truncates tool results when re-rendering history. Matters MORE when thinking is preserved, not less, because context is scarcer. The desktop log records a single grep returning 90k tokens with this unset and detonating the context.

#### `REASONING` — Reasoning mode

**Type:** select · **Default:** `on` · **Values:** `auto`, `on`, `off`

`-rea/--reasoning`. Was **hardcoded to 'on'** until 2026-09-15. 'auto' detects from the chat template, which for the pinned qwen3.8-safe-v2 template means thinking stays on anyway. 'off' is the only real way to get non-thinking replies out of this model without editing the template.

#### `REASONING_FORMAT` — Reasoning format

**Type:** select · **Default:** _(unset — engine default)_ · **Values:** `auto`, `none`, `deepseek`, `deepseek-legacy`

`--reasoning-format`. Where thoughts end up in the API response: **none** leaves them unparsed inside message.content, **deepseek** puts them in message.reasoning_content, **deepseek-legacy** does both (keeps the <think> tags in content AND fills reasoning_content). Stock is auto. Clients that render a collapsible 'thinking' block want deepseek; a client that shows raw content will display the entire chain of thought if this is none.

#### `REASONING_BUDGET_MESSAGE` — Budget-exhausted message

**Type:** text · **Default:** _(unset — engine default)_ · **Flag:** `--reasoning-budget-message`

`--reasoning-budget-message`, **new in b10985**. Text injected immediately before the end-of-thinking tag when REASONING_BUDGET runs out. Without it the model is cut off mid-thought and then has to answer from a truncated trace; with it you get a handoff line such as _"Budget reached - answer now with what you have."_ If you set a finite REASONING_BUDGET, set this too. Empty = none.

#### `KEEP` — Tokens to keep

**Type:** int · **Default:** _(unset — engine default)_ · **Flag:** `--keep` · **Values:** `0`, `-1`, `512`

`--keep`. Tokens kept from the start of the prompt when the context is shifted; stock 0, -1 = all. Only has an effect with CONTEXT_SHIFT on. Set it to cover the system prompt, or a shifted conversation loses its instructions first and nothing reports that it happened.

### llama.cpp Sampling

#### `TEMP` — Temperature

**Type:** float · **Default:** `1.0` · **Values:** `0.6`, `0.7`, `0.8`, `1.0`, `1.2`

Only a DEFAULT. Any OpenAI-compatible client that sends its own temperature overrides it. 1.0 with top_p 0.95 / top_k 20 is Qwen's recommended thinking-mode pairing.

#### `TOP_P` — top_p

**Type:** float · **Default:** `0.95` · **Values:** `0.8`, `0.9`, `0.95`, `1.0`

Nucleus sampling cutoff. Default only - a client's own value wins. Qwen recommends 0.95 for thinking mode.

#### `TOP_K` — top_k

**Type:** int · **Default:** `20` · **Values:** `0`, `20`, `40`, `64`, `100`

Consider only the k most likely tokens. Default only. Qwen recommends 20 for thinking mode.

#### `MIN_P` — min_p

**Type:** float · **Default:** `0.0` · **Values:** `0.0`, `0.01`, `0.05`, `0.1`

Drops tokens below this fraction of the top token's probability. 0.0 disables. Default only.

#### `SAMPLERS` — Sampler chain

**Type:** text · **Default:** _(unset — engine default)_ · **Flag:** `--samplers`

`--samplers`, ';'-separated and **order matters**. Stock chain in b10985 is `penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature`. Empty = that stock chain. Anything you omit here is disabled no matter what its own parameter says, which is the usual reason a sampler 'does nothing'.

#### `SEED` — RNG seed

**Type:** int · **Default:** _(unset — engine default)_ · **Flag:** `--seed` · **Values:** `-1`, `0`, `42`

`-s/--seed`. -1 (stock) = random per request. Pin it only for reproducing a specific output; a fixed seed on a shared server makes every client's generation correlated.

#### `TOP_N_SIGMA` — top_n_sigma

**Type:** float · **Default:** _(unset — engine default)_ · **Flag:** `--top-nsigma` · **Values:** `-1`, `1.0`

`--top-nsigma`, stock -1.0 = disabled. Keeps tokens within N standard deviations of the top logit. Sits before top_k in the stock chain.

#### `TYPICAL_P` — typical_p

**Type:** float · **Default:** _(unset — engine default)_ · **Flag:** `--typical` · **Values:** `1.0`, `0.95`

`--typical`, stock 1.0 = disabled. Locally typical sampling.

#### `REPEAT_PENALTY` — repeat_penalty

**Type:** float · **Default:** _(unset — engine default)_ · **Flag:** `--repeat-penalty` · **Values:** `1.0`, `1.05`, `1.1`

`--repeat-penalty`, stock 1.0 = disabled. Qwen thinking mode does not want this: a repetition penalty applied across a long chain of thought punishes the model for restating its own premises, which is exactly what reasoning does.

#### `REPEAT_LAST_N` — repeat_last_n

**Type:** int · **Default:** _(unset — engine default)_ · **Flag:** `--repeat-last-n` · **Values:** `0`, `64`, `256`

`--repeat-last-n`, stock 64, 0 = disabled. Window the repetition penalty looks back over.

#### `PRESENCE_PENALTY` — presence_penalty

**Type:** float · **Default:** _(unset — engine default)_ · **Flag:** `--presence-penalty` · **Values:** `0.0`, `0.5`

`--presence-penalty`, stock 0.0 = disabled. Default only; a client sending its own value wins.

#### `FREQUENCY_PENALTY` — frequency_penalty

**Type:** float · **Default:** _(unset — engine default)_ · **Flag:** `--frequency-penalty` · **Values:** `0.0`, `0.5`

`--frequency-penalty`, stock 0.0 = disabled. Default only.

#### `DRY_MULTIPLIER` — DRY multiplier

**Type:** float · **Default:** _(unset — engine default)_ · **Flag:** `--dry-multiplier` · **Values:** `0.0`, `0.8`

`--dry-multiplier`, stock 0.0 = disabled. DRY suppresses verbatim repetition of sequences rather than of single tokens, which makes it far safer on reasoning output than repeat_penalty. It is second in the stock sampler chain.

#### `DRY_BASE` — DRY base

**Type:** float · **Default:** _(unset — engine default)_ · **Flag:** `--dry-base` · **Values:** `1.75`

`--dry-base`, stock 1.75. Only read when DRY_MULTIPLIER > 0.

#### `DRY_ALLOWED_LENGTH` — DRY allowed length

**Type:** int · **Default:** _(unset — engine default)_ · **Flag:** `--dry-allowed-length` · **Values:** `2`

`--dry-allowed-length`, stock 2. Repeats shorter than this are not penalised. On code output raise it: indentation and closing braces are legitimate short repeats.

#### `DRY_PENALTY_LAST_N` — DRY window

**Type:** int · **Default:** _(unset — engine default)_ · **Flag:** `--dry-penalty-last-n` · **Values:** `0`, `64`, `512`

`--dry-penalty-last-n`, stock 64, 0 = disabled.

#### `XTC_PROBABILITY` — XTC probability

**Type:** float · **Default:** _(unset — engine default)_ · **Flag:** `--xtc-probability` · **Values:** `0.0`, `0.5`

`--xtc-probability`, stock 0.0 = disabled. Exclude Top Choices drops high-probability tokens to raise variety. Directly opposed to what you want from a coding or tool-calling model.

#### `XTC_THRESHOLD` — XTC threshold

**Type:** float · **Default:** _(unset — engine default)_ · **Flag:** `--xtc-threshold` · **Values:** `0.1`

`--xtc-threshold`, stock 0.1, 1.0 = disabled.

#### `ADAPTIVE_TARGET` — adaptive-p target

**Type:** float · **Default:** _(unset — engine default)_ · **Flag:** `--adaptive-target` · **Values:** `-1`, `0.1`

`--adaptive-target`, **new in b10985** (upstream PR 17927). adaptive-p selects tokens near this probability and adapts over time; valid 0.0 to 1.0, negative = disabled (stock -1.0). New enough that there are no numbers for it on this box.

#### `ADAPTIVE_DECAY` — adaptive-p decay

**Type:** float · **Default:** _(unset — engine default)_ · **Flag:** `--adaptive-decay` · **Values:** `0.90`

`--adaptive-decay`, stock 0.90, valid 0.0-0.99. Lower is more reactive, higher more stable. Only read when ADAPTIVE_TARGET is enabled.

#### `DYNATEMP_RANGE` — dynatemp range

**Type:** float · **Default:** _(unset — engine default)_ · **Flag:** `--dynatemp-range` · **Values:** `0.0`, `0.5`

`--dynatemp-range`, stock 0.0 = disabled. Varies temperature with entropy around TEMP.

#### `DYNATEMP_EXP` — dynatemp exponent

**Type:** float · **Default:** _(unset — engine default)_ · **Flag:** `--dynatemp-exp` · **Values:** `1.0`

`--dynatemp-exp`, stock 1.0. Only read when DYNATEMP_RANGE > 0.

#### `MIROSTAT` — Mirostat

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--mirostat` · **Values:** `0`, `1`, `2`

`--mirostat`, stock 0 = disabled. **1 or 2 makes top_k, top_p and typical_p be ignored entirely** - the panel will still show those values and they will still be sent, and none of them will do anything. Do not enable it and then tune top_p.

#### `MIROSTAT_LR` — Mirostat eta

**Type:** float · **Default:** _(unset — engine default)_ · **Flag:** `--mirostat-lr` · **Values:** `0.1`

`--mirostat-lr`, stock 0.10. Only read when MIROSTAT is 1 or 2.

#### `MIROSTAT_ENT` — Mirostat tau

**Type:** float · **Default:** _(unset — engine default)_ · **Flag:** `--mirostat-ent` · **Values:** `5.0`

`--mirostat-ent`, stock 5.00. Only read when MIROSTAT is 1 or 2.

### llama.cpp Safety

#### `RAM_FLOOR_MB` — Host RAM floor

**Type:** int · **Unit:** MB · **Default:** `512` · **Values:** `512`, `1024`, `2048`, `3072`, `4096`, `6144`

The launch script kills llama-server if MemAvailable drops below this, because a Linux OOM on a swap-thrashing 30 GB host does not reliably kill the offender before the machine locks up. 0 disables it. **Disabling is not advised:** it was set to 0 on 2026-09-04 and the box took a real global kernel OOM 1h43m later (llama-server killed, 7.8 GB of 8.19 GB swap consumed).

### llama.cpp Server

#### `PORT` — API port

**Type:** int · **Default:** `8081` · **Values:** `8081`, `8083`

Inference API port. Binds 0.0.0.0 and sits OUTSIDE Caddy with no authentication.

#### `MODEL` — Model file

**Type:** text · **Default:** `/home/admin/models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf`

Path to the GGUF served. Changing it needs a restart and a full reload.

#### `TEMPLATE_SRC` — Chat template

**Type:** select · **Default:** `/home/admin/models/qwen3.8-safe-v2.jinja`

The Jinja chat template, from `~/models/*.jinja`. Download one on the Download tab or paste one on the Templates tab, then pick it here.

The launch script SHA-pins the template and **aborts** on a mismatch. The panel writes the matching pin (`TEMPLATE_SHA256`) automatically whenever you save, so the check keeps protecting you without pinning one specific file. **Edit a selected template on disk and the next launch will abort** - re-save in the panel to re-pin it.

#### `HOST` — Bind address

**Type:** select · **Default:** `0.0.0.0` · **Values:** `0.0.0.0`, `127.0.0.1`

`--host`. Was **hardcoded to 0.0.0.0** until 2026-09-15, which is how port 8081 came to be reachable from the whole network with no authentication in front of it - Caddy only fronts the panel. Setting this to **127.0.0.1** is the one-line fix if you do not need remote inference; if you do, set API_KEY. Needs a restart, and will cut off any remote client immediately.

#### `API_KEY` — API key

**Type:** text · **Default:** _(unset — engine default)_

`--api-key`, comma-separated for several. **Empty means port 8081 accepts anything that can reach it.** That port bypasses Caddy entirely, so the panel's basicauth does not protect it and neither does anything else on this box. Setting a key here is the only authentication the inference API has. Note it is stored in cleartext in params.env and lands in the configuration backup tarball, which is mode 600 - keep it that way.

#### `TIMEOUT` — Request timeout

**Type:** int · **Unit:** s · **Default:** _(unset — engine default)_ · **Flag:** `--timeout` · **Values:** `600`, `3600`, `7200`

`-to/--timeout`, stock 3600. Server read/write timeout. A single deep-context request on this box can prefill for minutes before the first token; a client-side timeout shorter than this is the usual cause of a 'hang' that the server log shows completing normally.

#### `SSE_PING_INTERVAL` — SSE ping interval

**Type:** int · **Unit:** s · **Default:** _(unset — engine default)_ · **Flag:** `--sse-ping-interval` · **Values:** `-1`, `15`, `30`

`--sse-ping-interval`, stock 30, -1 = disabled. Keepalive on streaming responses. Raise or disable it only if a proxy is mangling the stream.

#### `SLEEP_IDLE_SECONDS` — Sleep when idle

**Type:** int · **Unit:** s · **Default:** _(unset — engine default)_ · **Flag:** `--sleep-idle-seconds` · **Values:** `-1`, `300`, `900`, `1800`

`--sleep-idle-seconds`, stock -1 = disabled. Puts the server to sleep after N idle seconds. Interesting on this box specifically: a 291 W card in a 300 W Dell T40 is the working theory for the silent hard locks, and an idle server that has released the GPU is one fewer thing holding the rail up. The cost is that the next request pays a wake-up, and no measurement of that cost has been taken here yet.

#### `WEBUI` — Built-in web UI

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--webui / --no-webui` · **Values:** `on`, `off`

`--ui/--webui` / `--no-webui`, stock enabled. llama.cpp's own chat UI on port 8081. It is served on the same unauthenticated port as the API, so 'off' is a reasonable default here - this panel is the intended UI.

#### `PROPS` — Allow POST /props

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--props` · **Values:** `on`, `off`

`--props`, stock **disabled**. Lets any client change global server properties over HTTP. On an unauthenticated port that means anyone who can reach 8081 can reconfigure the server out from under this panel. Leave it off unless API_KEY is set.

#### `SLOTS` — Slots endpoint

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--slots / --no-slots` · **Values:** `on`, `off`

`--slots/--no-slots`, stock enabled. Exposes /slots, which the panel's own diagnostics read. Turning it off blinds this panel's slot view; it also stops leaking prompt content to anyone who can reach 8081.

#### `ALIAS` — Model alias

**Type:** text · **Default:** _(unset — engine default)_ · **Flag:** `--alias`

`-a/--alias`, comma-separated. The model name the API reports and that OpenAI-compatible clients must send. Empty = the GGUF filename, which for this model is the 96-character `Qwen3.8-27B-TurboFCFusion-735-882-...-IQ4_XS`. Setting a short alias is usually less trouble than making every client quote that.

#### `LOG_VERBOSITY` — Log verbosity

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--log-verbosity` · **Values:** `0`, `1`, `2`, `3`, `4`, `5`

`-lv/--log-verbosity`, stock 3 (info). 0 generic, 1 error, 2 warning, 3 info, 4 trace, 5 debug. 4 and 5 write a great deal into `/dev/shm/llama_qwen38`, which is a tmpfs - it costs host RAM, and host RAM is what the watchdog kills the server over.

## stable-diffusion.cpp instances

One `sd-server` per instance, image generation jobs via its async API.

| Group | Settings |
|---|---|
| [Backend](#stable-diffusioncpp-backend) | 1 |
| [Model files](#stable-diffusioncpp-model-files) | 8 |
| [Placement & memory](#stable-diffusioncpp-placement--memory) | 9 |
| [Generation defaults](#stable-diffusioncpp-generation-defaults) | 9 |
| [Server](#stable-diffusioncpp-server) | 4 |

### stable-diffusion.cpp Backend

#### `BACKEND` — Compute backend

**Type:** select · **Default:** `vulkan` · **Flag:** `(which build)` · **Values:** `vulkan`, `rocm`, `cpu`

Which kind of program runs the model. **vulkan** works on both graphics cards (the 7900 XTX and the RTX 2060) - pick this. **rocm** is AMD-only. **cpu** uses no graphics card and is very slow (minutes per image).

### stable-diffusion.cpp Model files

#### `SD_PRESET` — Preset

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `SD_PRESET` · **Values:** `qwen-image-2.1`

The model recipe last applied. Use the Model preset card above to fill everything in for you.

#### `SD_DIFFUSION_MODEL` — Diffusion model

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--diffusion-model` · **Values:** `/home/admin/models/sd/Qwen3VL-8B-Instruct-Q4_K_M.gguf`, `/home/admin/models/sd/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf`, `/home/admin/models/sd/qwen_image_2.1_vae_bf16.safetensors`, `/home/admin/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-LOW-MTP-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf`, `/home/admin/models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00002-of-00002.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-MTP-draft.gguf`, `/home/admin/models/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf`, `/home/admin/models/mmproj-F16.gguf`, `/home/admin/models/mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf`, `/home/admin/models/mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf`, `/home/admin/models/qwen-image-2.1-Q5_K_M.gguf`

The main image model - the part that actually draws. For Qwen-Image 2.1 this is the qwen-image-2.1 .gguf file.

#### `SD_MODEL` — Full checkpoint

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `-m` · **Values:** `/home/admin/models/sd/Qwen3VL-8B-Instruct-Q4_K_M.gguf`, `/home/admin/models/sd/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf`, `/home/admin/models/sd/qwen_image_2.1_vae_bf16.safetensors`, `/home/admin/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-LOW-MTP-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf`, `/home/admin/models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00002-of-00002.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-MTP-draft.gguf`, `/home/admin/models/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf`, `/home/admin/models/mmproj-F16.gguf`, `/home/admin/models/mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf`, `/home/admin/models/mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf`, `/home/admin/models/qwen-image-2.1-Q5_K_M.gguf`

Only for older all-in-one model files (Stable Diffusion 1.5 / SDXL). Leave empty for Qwen-Image.

#### `SD_LLM` — Text encoder (LLM)

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--llm` · **Values:** `/home/admin/models/sd/Qwen3VL-8B-Instruct-Q4_K_M.gguf`, `/home/admin/models/sd/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf`, `/home/admin/models/sd/qwen_image_2.1_vae_bf16.safetensors`, `/home/admin/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-LOW-MTP-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf`, `/home/admin/models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00002-of-00002.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-MTP-draft.gguf`, `/home/admin/models/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf`, `/home/admin/models/mmproj-F16.gguf`, `/home/admin/models/mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf`, `/home/admin/models/mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf`, `/home/admin/models/qwen-image-2.1-Q5_K_M.gguf`

The text encoder: a language model that reads your prompt and explains it to the image model. Qwen-Image 2.1 needs Qwen3-VL-8B.

#### `SD_LLM_VISION` — Text encoder vision (mmproj)

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--llm_vision` · **Values:** `/home/admin/models/sd/Qwen3VL-8B-Instruct-Q4_K_M.gguf`, `/home/admin/models/sd/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf`, `/home/admin/models/sd/qwen_image_2.1_vae_bf16.safetensors`, `/home/admin/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-LOW-MTP-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf`, `/home/admin/models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00002-of-00002.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-MTP-draft.gguf`, `/home/admin/models/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf`, `/home/admin/models/mmproj-F16.gguf`, `/home/admin/models/mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf`, `/home/admin/models/mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf`, `/home/admin/models/qwen-image-2.1-Q5_K_M.gguf`

Lets the text encoder look at pictures. Only needed for editing an existing image; leave empty to save memory.

#### `SD_VAE` — VAE

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--vae` · **Values:** `/home/admin/models/sd/Qwen3VL-8B-Instruct-Q4_K_M.gguf`, `/home/admin/models/sd/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf`, `/home/admin/models/sd/qwen_image_2.1_vae_bf16.safetensors`, `/home/admin/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-LOW-MTP-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf`, `/home/admin/models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00002-of-00002.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-MTP-draft.gguf`, `/home/admin/models/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf`, `/home/admin/models/mmproj-F16.gguf`, `/home/admin/models/mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf`, `/home/admin/models/mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf`, `/home/admin/models/qwen-image-2.1-Q5_K_M.gguf`

Turns the model's internal result into an actual picture. Must be the one made for this model - a wrong VAE gives garbage colours.

#### `SD_CLIP_L` — CLIP-L

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--clip_l` · **Values:** `/home/admin/models/sd/Qwen3VL-8B-Instruct-Q4_K_M.gguf`, `/home/admin/models/sd/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf`, `/home/admin/models/sd/qwen_image_2.1_vae_bf16.safetensors`, `/home/admin/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-LOW-MTP-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf`, `/home/admin/models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00002-of-00002.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-MTP-draft.gguf`, `/home/admin/models/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf`, `/home/admin/models/mmproj-F16.gguf`, `/home/admin/models/mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf`, `/home/admin/models/mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf`, `/home/admin/models/qwen-image-2.1-Q5_K_M.gguf`

Extra text encoder for Flux 1 and SD3 models. Not used by Qwen-Image.

#### `SD_T5XXL` — T5-XXL

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--t5xxl` · **Values:** `/home/admin/models/sd/Qwen3VL-8B-Instruct-Q4_K_M.gguf`, `/home/admin/models/sd/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf`, `/home/admin/models/sd/qwen_image_2.1_vae_bf16.safetensors`, `/home/admin/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-LOW-MTP-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf`, `/home/admin/models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00002-of-00002.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-MTP-draft.gguf`, `/home/admin/models/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf`, `/home/admin/models/mmproj-F16.gguf`, `/home/admin/models/mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf`, `/home/admin/models/mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf`, `/home/admin/models/qwen-image-2.1-Q5_K_M.gguf`

Another extra text encoder for Flux 1 and SD3. Not used by Qwen-Image.

### stable-diffusion.cpp Placement & memory

#### `SD_TE_ON` — Text encoder on

**Type:** select · **Default:** `cpu` · **Flag:** `--backend te=...` · **Values:** `cpu`, `gpu`, `second`

Where the text encoder runs. **cpu**: saves graphics memory, costs a few seconds per image - best on the 6 GB RTX 2060. **gpu**: fastest, needs memory. **second**: on this instance's second graphics card.

#### `SD_VAE_ON` — VAE on

**Type:** select · **Default:** `gpu` · **Flag:** `--backend vae=...` · **Values:** `gpu`, `cpu`

Where the picture is decoded at the end. **gpu** is much faster. Pick **cpu** only if the last step runs out of graphics memory.

#### `SD_OFFLOAD` — Offload weights to RAM

**Type:** bool · **Default:** `1` · **Flag:** `--offload-to-cpu`

Keeps the models in normal RAM and moves each part onto the graphics card only while it works. Needed on small cards; a bit slower. Leave on.

#### `SD_DIFFUSION_FA` — Flash attention (diffusion)

**Type:** bool · **Default:** `1` · **Flag:** `--diffusion-fa`

A faster, less memory-hungry way of doing the model's maths. Leave on. Turn off only if pictures come out black.

#### `SD_VAE_TILING` — VAE tiling

**Type:** bool · **Default:** `0` · **Flag:** `--vae-tiling`

Decodes the final picture in pieces so big images fit on a small card. Turn on for the 6 GB card or if the last step runs out of memory.

#### `SD_MMAP` — Memory-map files

**Type:** bool · **Default:** `1` · **Flag:** `--mmap`

Reads model files straight from disk as needed instead of copying them into memory first. Faster start, less RAM. Leave on.

#### `SD_MAX_VRAM` — VRAM budget

**Type:** text · **Unit:** GiB · **Default:** _(unset — engine default)_ · **Flag:** `--max-vram`

Caps how much graphics memory it may use, in GB (e.g. 5). Empty = use whatever is free. Set it if something else shares the card.

#### `SD_THREADS` — CPU threads

**Type:** int · **Default:** `-1` · **Flag:** `-t`

How many CPU cores to use for work done on the CPU. -1 = automatic (all real cores).

#### `RAM_FLOOR_MB` — Host RAM floor

**Type:** int · **Unit:** MB · **Default:** `2048` · **Flag:** `RAM_FLOOR_MB`

Safety net: if free RAM drops below this, the server is stopped before the whole machine freezes. Do not set it lower than 2048.

### stable-diffusion.cpp Generation defaults

#### `SD_WIDTH` — Width

**Type:** int · **Unit:** px · **Default:** `1024` · **Flag:** `-W`

Picture width in pixels when an app doesn't say. Must be a multiple of 32. Bigger = slower and more memory.

#### `SD_HEIGHT` — Height

**Type:** int · **Unit:** px · **Default:** `1024` · **Flag:** `-H`

Picture height in pixels when an app doesn't say. Must be a multiple of 32.

#### `SD_STEPS` — Steps

**Type:** int · **Default:** `20` · **Flag:** `--steps`

How many times the model refines the picture. More steps = more detail but slower (time grows in a straight line). 20-30 is typical.

#### `SD_CFG` — CFG scale

**Type:** float · **Default:** `6.0` · **Flag:** `--cfg-scale`

How strictly the picture follows your prompt. Too low = ignores you, too high = harsh, over-cooked images. Qwen-Image 2.1 recommends about 6.

#### `SD_SAMPLER` — Sampler

**Type:** select · **Default:** `euler` · **Flag:** `--sampling-method` · **Values:** `euler`, `euler_a`, `heun`, `dpm2`, `dpm++2s_a`, `dpm++2m`, `dpm++2mv2`, `ipndm`, `ipndm_v`, `lcm`, `ddim_trailing`, `tcd`, `res_multistep`, `res_2s`, `er_sde`, `euler_cfg_pp`, `euler_a_cfg_pp`, `euler_ge`, `dpm++2m_sde`, `dpm++2m_sde_bt`, `lms`

The method used to refine the picture step by step. **euler** is the safe choice for Qwen-Image; others can look different but rarely better.

#### `SD_SCHEDULER` — Scheduler

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--scheduler` · **Values:** `discrete`, `karras`, `exponential`, `ays`, `gits`, `sgm_uniform`, `simple`, `smoothstep`, `kl_optimal`, `lcm`, `bong_tangent`, `logit_normal`, `flux2`, `flux`, `beta`

How the refining is spread over the steps. Empty = the model's own choice, which is almost always right.

#### `SD_FLOW_SHIFT` — Flow shift

**Type:** text · **Default:** _(unset — engine default)_ · **Flag:** `--flow-shift`

Shifts effort between the rough shape and the fine detail. Empty = automatic, which is right for Qwen-Image 2.1.

#### `SD_SEED` — Seed

**Type:** int · **Default:** `-1` · **Flag:** `-s`

The random starting point. The same seed + same settings = the same picture. -1 = a new random picture each time.

#### `SD_NEGATIVE` — Negative prompt

**Type:** text · **Default:** _(unset — engine default)_ · **Flag:** `-n`

Things you do NOT want in the picture, e.g. 'blurry, text, watermark'. Used when an app sends none.

### stable-diffusion.cpp Server

#### `PORT` — Port

**Type:** int · **Default:** `8084` · **Flag:** `--listen-port`

The network port the image server listens on. The panel talks to it for you.

#### `HOST` — Listen address

**Type:** select · **Default:** `127.0.0.1` · **Flag:** `--listen-ip` · **Values:** `127.0.0.1`, `0.0.0.0`

Who may connect. **127.0.0.1** = only this machine (safe; use the panel's Generate card). **0.0.0.0** = anyone on the network, with NO password.

#### `SD_LOG_LEVEL` — Log level

**Type:** select · **Default:** `info` · **Flag:** `--log-level` · **Values:** `info`, `verbose`, `debug`, `warn`, `error`

How chatty the log is. **info** is right; **debug** only to chase a problem.

#### `SD_EXTRA` — Extra arguments

**Type:** text · **Default:** _(unset — engine default)_ · **Flag:** `SD_EXTRA`

Any other option, typed by hand. Easier: tick them in 'All other options from this build' below.

## audio.cpp instances

One `audiocpp_server` per instance, one device per server. These are the server-level settings; which models an instance serves, and each model's **load**, **session** and **default request** options, are edited on the Parameters tab and stored in `instances/<id>/audio-models.json`. Those per-model option forms are generated from the installed audio.cpp build's `model_specs/*.json` (type, range, default and upstream's description), so they always match the build. Load and session options are written to the server config as `<family>.<name>`; request options keep their plain names.

| Group | Settings |
|---|---|
| [Backend](#audiocpp-backend) | 2 |
| [Memory & residency](#audiocpp-memory--residency) | 5 |
| [Server](#audiocpp-server) | 6 |

### audio.cpp Backend

#### `BACKEND` — Compute backend

**Type:** select · **Default:** `vulkan` · **Flag:** `--backend` · **Values:** `vulkan`, `cpu`

What runs the models. **vulkan** uses the graphics card picked for this instance. **cpu** uses no graphics card: small speech models (Kokoro, Kitten, Piper, Supertonic) are fast enough on the CPU, music and big voice-cloning models are not. Upstream tunes audio.cpp for CUDA, so on Vulkan some models are slower or unsupported.

#### `AC_THREADS` — CPU threads

**Type:** int · **Default:** `4` · **Flag:** `--threads`

How many CPU cores the CPU parts use. This box has 8; leave some for the chat model and the panel.

### audio.cpp Memory & residency

#### `AC_MAX_LOADED` — Models kept loaded

**Type:** int · **Default:** `1` · **Flag:** `--max-loaded-models`

How many models may sit in memory at once. When one more is needed, the least recently used idle one is unloaded first. **1** means one at a time, the safe choice while the chat model owns most of the card. **0** = no limit.

#### `AC_IDLE_UNLOAD_S` — Unload when idle after

**Type:** int · **Unit:** s · **Default:** `600` · **Flag:** `--idle-unload-ms`

Frees every loaded model after this many seconds without a request, handing the memory back to the chat model. The next request loads it again (seconds for speech models, longer for music). **0** = never.

#### `AC_MIN_FREE_MB` — Keep free

**Type:** int · **Unit:** MiB · **Default:** `1024` · **Flag:** `--min-free-memory-mb`

Refuse to load a model unless this much host RAM and graphics memory is still free afterwards, so a load fails cleanly instead of starving the chat model. **0** = no check.

#### `AC_LAZY` — Load on first use

**Type:** bool · **Default:** `1`

Load each model only when it is first asked for, instead of all at start. Keeps start-up fast and memory low.

#### `RAM_FLOOR_MB` — Host RAM floor

**Type:** int · **Unit:** MB · **Default:** `2048`

The launcher kills audiocpp_server if available host RAM drops below this. The host has hard-locked from running out of RAM.

### audio.cpp Server

#### `AC_BUSY_TIMEOUT_S` — Busy timeout

**Type:** int · **Unit:** s · **Default:** `300` · **Flag:** `--busy-timeout-ms`

A request waiting this long for a busy model fails instead of queueing forever. Long music jobs hold the model for minutes, so raise it if requests pile up behind them. **0** = wait forever.

#### `AC_UI` — Built-in web UI

**Type:** bool · **Default:** `1` · **Flag:** `--ui / --no-ui`

audio.cpp's own browser UI (Arena, voice library, microphone input) on the instance's port. It has no password: keep the listen address on 127.0.0.1 and reach it through an SSH tunnel.

#### `AC_LOG` — Framework log

**Type:** bool · **Default:** `1` · **Flag:** `--log`

Write audio.cpp's own log to the engine log (Logs tab).

#### `PORT` — Port

**Type:** int · **Default:** `8085` · **Flag:** `--port`

Where audiocpp_server listens.

#### `HOST` — Listen address

**Type:** select · **Default:** `127.0.0.1` · **Flag:** `--host` · **Values:** `127.0.0.1`, `0.0.0.0`

audiocpp_server has no authentication: keep 127.0.0.1 and use it through the panel.

#### `AC_EXTRA` — Extra arguments

**Type:** text · **Default:** _(unset — engine default)_

Anything else audiocpp_server accepts, passed as-is (shell-quoted).

## Camelid instances

One `camelid serve` per instance (github.com/timtoole02/Camelid): a Rust GGUF chat engine with its own web UI and an OpenAI-compatible API. CPU, or an NVIDIA card with `cuda`; it has no Vulkan or ROCm, so AMD cards are refused. Models come from its curated catalog (pulled from the Parameters tab) or any GGUF it accepts. The API key is stored in a 0600 file and passed with `--api-key-file`.

| Group | Settings |
|---|---|
| [Backend](#camelid-backend) | 3 |
| [Model](#camelid-model) | 3 |
| [Generation](#camelid-generation) | 7 |
| [Server](#camelid-server) | 5 |

### Camelid Backend

#### `BACKEND` — Compute backend

**Type:** select · **Default:** `cpu` · **Flag:** `--gpu off / on` · **Values:** `cpu`, `cuda`

**cpu** runs everywhere and is the only choice on AMD cards: Camelid has no Vulkan or ROCm. **cuda** uses an NVIDIA card (the Linux build ships its own CUDA runtime; the NVIDIA driver must be installed).

#### `CM_THREADS` — CPU threads

**Type:** int · **Default:** `6` · **Flag:** `--threads`

Worker threads for the CPU path. This box has 8 cores; leave some for the other servers.

#### `RAM_FLOOR_MB` — Host RAM floor

**Type:** int · **Unit:** MB · **Default:** `2048`

The launcher stops Camelid if free host RAM falls below this. The host has hard-locked from running out of RAM.

### Camelid Model

#### `CM_MODEL` — Model file

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--model` · **Values:** `/home/admin/camelid/models/Qwen3-0.6B-Q8_0.gguf`, `/home/admin/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-LOW-MTP-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf`, `/home/admin/models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00002-of-00002.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-MTP-draft.gguf`, `/home/admin/models/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf`, `/home/admin/models/qwen-image-2.1-Q5_K_M.gguf`, `/home/admin/models/sd/Qwen3VL-8B-Instruct-Q4_K_M.gguf`

The GGUF to load at start. Camelid supports a curated list of models (the catalog below, with a fit check for this machine); other GGUFs may refuse to load. Files from ~/camelid/models and ~/models are offered.

#### `CM_MODELS_DIR` — Models folder

**Type:** text · **Default:** _(unset — engine default)_ · **Flag:** `--models-dir`

Where Camelid's catalog downloads go and what its own web UI lists. Empty = ~/camelid/models.

#### `CM_KV_QUANT` — KV cache precision

**Type:** select · **Default:** `f16` · **Flag:** `--kv-quant` · **Values:** `f16`, `q8_0`, `q4_0`, `fp8_e4m3`, `fp8_e5m2`

Memory used per token of context. **f16** exact; **q8_0** and the **fp8** formats halve it; **q4_0** quarters it. On the CUDA path only f16 and q8_0 are honoured.

### Camelid Generation

#### `CM_THINKING` — Thinking on by default

**Type:** bool · **Default:** `0` · **Flag:** `--enable-thinking`

Qwen3 / Gemma 4 think before answering unless a request says otherwise. Better answers, slower replies.

#### `CM_DETERMINISTIC` — Deterministic

**Type:** bool · **Default:** `0` · **Flag:** `--deterministic`

Same input gives bit-identical output (order-stable CPU path, GPU off). Slower; for testing.

#### `CM_SPEC` — Speculative decoding

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--spec-decode` · **Values:** `ngram`, `draft`

Faster replies with identical output. **ngram** guesses ahead from the prompt (no extra model). **draft** uses a small model with the same tokenizer (set below). Empty = off.

#### `CM_SPEC_DRAFT` — Draft model

**Type:** select · **Default:** _(unset — engine default)_ · **Flag:** `--spec-draft-model` · **Values:** `/home/admin/camelid/models/Qwen3-0.6B-Q8_0.gguf`, `/home/admin/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-LOW-MTP-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf`, `/home/admin/models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00002-of-00002.gguf`, `/home/admin/models/Qwen3.8-Flash-Next-Uncensored-MTP-draft.gguf`, `/home/admin/models/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf`, `/home/admin/models/qwen-image-2.1-Q5_K_M.gguf`, `/home/admin/models/sd/Qwen3VL-8B-Instruct-Q4_K_M.gguf`

Small GGUF for draft mode; must share the main model's tokenizer.

#### `CM_SPEC_TOKENS` — Draft tokens per round

**Type:** int · **Default:** `5` · **Flag:** `--spec-draft-tokens`

How far each guess runs ahead.

#### `CM_MAX_PROMPT` — Max prompt tokens

**Type:** int · **Default:** `131072` · **Flag:** `--max-prompt-tokens`

Longest prompt accepted.

#### `CM_MAX_GEN` — Max reply tokens

**Type:** int · **Default:** `8192` · **Flag:** `--max-generation-tokens`

Largest max_tokens a request may ask for.

### Camelid Server

#### `PORT` — Port

**Type:** int · **Default:** `8086` · **Flag:** `--addr`

Camelid's API and web UI.

#### `HOST` — Listen address

**Type:** select · **Default:** `127.0.0.1` · **Flag:** `--addr` · **Values:** `127.0.0.1`, `0.0.0.0`

**127.0.0.1**: this box only (reach the web UI through an SSH tunnel). **0.0.0.0**: the LAN; then an API key is required and traffic is unencrypted.

#### `CM_API_KEY` — API key

**Type:** text · **Default:** _(unset — engine default)_ · **Flag:** `--api-key-file`

Required for a LAN listener. Stored in a 0600 file, passed with --api-key-file so it never shows in the process list.

#### `CM_LAN_CHAT_ONLY` — LAN: chat only

**Type:** bool · **Default:** `0` · **Flag:** `--lan-chat-only`

On a LAN listener, expose only chat (no model changes, workspace or agents).

#### `CM_EXTRA` — Extra arguments

**Type:** text · **Default:** _(unset — engine default)_

Anything else `camelid serve` accepts.
