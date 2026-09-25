#!/usr/bin/env python3
"""
Plain-English tooltips (added 2026-09-23).

Upstream's --help text is written for people who already know the internals.
Every option the panel shows gets a sentence or two here in plain words: what
it does, and whether a normal setup should touch it. Options a future build
adds, which nobody has written text for yet, get a generated explanation from
the patterns at the bottom, plus upstream's own words.

LEAVE = the standard advice for an option most people never need.
"""
import re

LEAVE = " If you are not sure, leave it off - the default is what almost everyone uses."

# ---------------------------------------------------------------------------
# llama-server options that are not in the curated form
# ---------------------------------------------------------------------------
LLAMA = {
    "--cpu-mask": "Pins the model's CPU threads to specific CPU cores, written as a hex bitmask "
                  "(e.g. ff = first 8 cores). Only useful to keep the model off cores another "
                  "program needs. The model runs on the GPU here, so this barely matters." + LEAVE,
    "--cpu-range": "Same as the CPU mask, but written as a range like 0-5 (cores 0 to 5)." + LEAVE,
    "--cpu-strict": "1 = threads may ONLY run on the cores picked above; 0 = the picks are a "
                    "preference." + LEAVE,
    "--prio": "How much the operating system favours the model's threads: -1 low, 0 normal, "
              "1 medium, 2 high, 3 realtime. High/realtime can make the rest of the box "
              "sluggish." + LEAVE,
    "--poll": "0-100. How hard idle threads spin waiting for work. Higher = slightly lower "
              "latency but burns CPU (and heat) while waiting." + LEAVE,
    "--cpu-mask-batch": "Core pinning like the CPU mask, but only for the phase where the model "
                        "reads your prompt." + LEAVE,
    "--cpu-range-batch": "Core range like the CPU range, but only for reading the prompt." + LEAVE,
    "--cpu-strict-batch": "Strict pinning, but only for reading the prompt." + LEAVE,
    "--prio-batch": "Thread priority, but only for reading the prompt." + LEAVE,
    "--poll-batch": "Idle spinning, but only for reading the prompt." + LEAVE,
    "--perf": "Prints extra timing measurements into the log. Handy when chasing a slowdown, "
              "noise otherwise.",
    "--escape": "Turns text like \\n in the prompt into real line breaks. On by default and "
                "harmless.",
    "--defrag-thold": "Old setting for tidying the memory the conversation lives in. Upstream has "
                      "deprecated it and it does nothing useful now. Leave it off.",
    "--rpc": "Uses other computers on the network as extra GPUs (each running llama.cpp's "
             "rpc-server). Slow over normal networks, and unencrypted. Only for experiments.",
    "--numa": "For big servers with more than one CPU socket. This box has one CPU, so it does "
              "nothing here. Leave it off.",
    "--n-cpu-ffn": "Keeps part of the first N layers on the CPU instead of the GPU, to save "
                   "video memory. Makes the model slower. Only if the model does not fit.",
    "--check-tensors": "Checks every number in the model file for corruption while loading. "
                       "Slows loading down; use it once if you suspect a damaged download.",
    "--override-kv": "Changes a setting stored inside the model file, e.g. fixes a wrong "
                     "tokenizer flag, without editing the file. Only if a model card tells you "
                     "to.",
    "--lora": "Loads a LoRA: a small add-on file that changes the model's style or skills "
              "without replacing it. Point it at the .gguf LoRA file.",
    "--lora-scaled": "Loads a LoRA with a strength, e.g. file.gguf:0.5 for half strength.",
    "--control-vector": "Loads a control vector: a small file that nudges the model's "
                        "personality or tone (e.g. more cheerful). Experimental.",
    "--control-vector-scaled": "Loads a control vector with a strength, e.g. file.gguf:0.8.",
    "--control-vector-layer-range": "Which layers the control vector applies to, as two "
                                    "numbers (start end)." + LEAVE,
    "--docker-repo": "Downloads the model from Docker Hub instead of using a file. The panel "
                     "manages model files itself, so do not use this here.",
    "--hf-file": "Picks which file to fetch from a Hugging Face repo. The panel downloads "
                 "models on the Download tab instead." + LEAVE,
    "--log-disable": "Turns logging off completely. The panel reads the log for its statistics "
                     "and diagnostics, so leave this off.",
    "--log-jsonl": "Writes the log as machine-readable JSON lines. The panel's parsers expect "
                   "normal text; leave this off.",
    "--log-colors": "Coloured log output. Only matters in a terminal." + LEAVE,
    "--verbose": "Logs everything, including every token. Huge logs; only for a short "
                 "debugging session.",
    "--offline": "Never touches the internet; only uses files already on disk. Safe to turn "
                 "on, since the panel gives the server local files anyway.",
    "--log-prefix": "Adds a short tag in front of each log line." + LEAVE,
    "--log-timestamps": "Adds the time to each log line. Harmless and handy when reading logs "
                        "later.",
    "--sampler-seq": "The order the word-picking rules run in, as one letter per rule. Only "
                     "change it if you know why." + LEAVE,
    "--ignore-eos": "Makes the model keep writing after it would normally stop. Only for "
                    "benchmarks; in real use it produces rambling.",
    "--dry-sequence-breaker": "For the DRY anti-repetition rule: characters that 'reset' what "
                              "counts as a repeat (default: newline, colon, quote, star)."
                              + LEAVE,
    "--logit-bias": "Makes a specific token more or less likely, by its number, e.g. "
                    "15043+1. Needs you to know token numbers; rarely worth it.",
    "--grammar": "Forces the output to follow a strict pattern (a GBNF grammar), e.g. only "
                 "valid JSON. Applies to EVERY request, which breaks normal chat. Better set "
                 "per request by the app.",
    "--grammar-file": "Same as grammar, read from a file. Applies to every request." + LEAVE,
    "--json-schema": "Forces every answer to be JSON matching this schema. Breaks normal chat "
                     "for everyone else; apps should send it per request instead.",
    "--json-schema-file": "Same as JSON schema, read from a file." + LEAVE,
    "--spec-draft-threads": "CPU threads for the separate draft model. This setup uses the "
                            "built-in MTP head instead of a draft model, so it does nothing "
                            "here.",
    "--spec-draft-threads-batch": "Prompt-reading threads for a separate draft model. Not used "
                                  "with the built-in MTP head.",
    "--spec-draft-override-tensor": "Places parts of a separate draft model on specific "
                                    "devices. Not used with the built-in MTP head.",
    "--spec-draft-cpu-moe": "Keeps a separate draft model's expert weights on the CPU. Not "
                            "used with the built-in MTP head.",
    "--spec-draft-n-cpu-moe": "Keeps the first N layers' expert weights of a separate draft "
                              "model on the CPU. Not used with the built-in MTP head.",
    "--spec-synth-len": "Fakes how well speculative decoding guesses, to measure the best "
                        "case. Benchmarking only - never for real use.",
    "--spec-synth-rates": "Fakes guess-acceptance rates per position. Benchmarking only.",
    "--lookup-cache-static": "A file of common word sequences used to guess ahead (lookup "
                             "decoding). Only with the lookup speculative mode." + LEAVE,
    "--lookup-cache-dynamic": "Like the static lookup cache, but it learns from what the model "
                              "writes." + LEAVE,
    "--reverse-prompt": "Stops writing when this text appears. For the old interactive mode; "
                        "apps normally send their own stop words.",
    "--special": "Shows special control tokens (like <|im_end|>) in the output text. Only for "
                 "debugging templates.",
    "--spm-infill": "Changes the order used for fill-in-the-middle code completion. Only if a "
                    "code model's card asks for it.",
    "--pooling": "For embedding models only: how a whole text is squeezed into one vector "
                 "(mean, cls, last...). Chat models ignore it.",
    "--mmproj-url": "Downloads the image-understanding (vision) file from a URL. The panel "
                    "picks a local file instead." + LEAVE,
    "--video-fps": "When a video is sent to a vision model: how many frames per second it "
                   "looks at. More frames = better understanding but many more tokens.",
    "--video-timestamp-interval": "How often (in ms) time labels are inserted between video "
                                  "frames for the model." + LEAVE,
    "--video-ffmpeg-dir": "Where ffmpeg lives, for reading videos. Only if ffmpeg is not "
                          "installed normally.",
    "--tags": "Labels for the model, shown to clients. Purely cosmetic.",
    "--embd-normalize": "For embedding models only: how vectors are scaled. 2 (the default) "
                        "is what almost every tool expects.",
    "--reuse-port": "Lets two servers listen on the same port. Here that would hide a port "
                    "clash between instances instead of reporting it. Leave it off.",
    "--path": "Serves files from this folder over the API port. Anyone who can reach the port "
              "can read them. Leave it off.",
    "--cors-origins": "Which websites may call this server from a browser. * = any site. "
                      "Only matters for browser apps.",
    "--cors-methods": "Which HTTP methods browser apps may use." + LEAVE,
    "--cors-headers": "Which headers browser apps may send." + LEAVE,
    "--cors-credentials": "Whether browsers may send cookies/logins along. With 'any site' "
                          "allowed, this lets any web page you visit use the server as you. "
                          "Keep it off.",
    "--api-prefix": "Moves the API under a sub-path like /llm. Clients, the panel and the "
                    "optimizer all expect the root path; changing it breaks them.",
    "--ui-config": "Default settings for llama.cpp's own built-in web chat page, as JSON.",
    "--ui-config-file": "Same as UI config, read from a file.",
    "--ui-mcp-proxy": "Lets the built-in web chat reach MCP tool servers through this server. "
                      "Experimental and a security risk on an open port.",
    "--tools": "Gives the model built-in tools like reading files and RUNNING SHELL COMMANDS on "
               "this machine. Anyone who can reach the port could use them. Dangerous here, "
               "because the API port has no password.",
    "--tools-runtime": "Runs those built-in tools inside a Docker/Podman container instead of "
                       "directly on the machine. Safer, if tools are on at all.",
    "--mcp-servers-config": "A file listing external tool (MCP) servers the model may call. "
                            "Experimental; a security risk on an open port.",
    "--mcp-servers-json": "The same list of tool servers, written inline." + LEAVE,
    "--agent": "Turns on ALL built-in tools (including shell commands) plus a web proxy. "
               "Dangerous on a port without a password. Leave it off.",
    "--embedding": "Makes the server ONLY produce embeddings (text-to-vector), not chat. Only "
                   "for a dedicated embedding model.",
    "--rerank": "Adds a search-result reranking endpoint. Only for reranker models.",
    "--ssl-key-file": "Private key for serving HTTPS directly. Caddy already does HTTPS for "
                      "the panel; only for exposing the API itself securely.",
    "--ssl-cert-file": "Certificate for serving HTTPS directly (pairs with the key file).",
    "--media-path": "A folder whose images/videos requests may refer to as file:// links. "
                    "Anyone who can reach the port could read them." + LEAVE,
    "--models-dir": "Router mode: one server that loads different models on demand from this "
                    "folder. A different way of working from the panel's instances.",
    "--models-preset": "Router mode: settings per model." + LEAVE,
    "--models-max": "Router mode: how many models may be loaded at once." + LEAVE,
    "--models-autoload": "Router mode: load a model automatically when a request asks for it."
                         + LEAVE,
    "--reasoning-effort": "How hard a thinking model thinks by default (low / medium / high). "
                          "The form's REASONING_EFFORT already sets this through the template.",
    "--chat-template": "Replaces the model's chat format with a named built-in one. The form's "
                       "template setting is the better way to do this.",
    "--skip-chat-parsing": "Stops the server separating 'thinking' and tool calls from the "
                           "answer; everything arrives as plain text. Breaks agents that "
                           "use tools.",
    "--prefill-assistant": "If a request ends with a half-written assistant message, continue "
                           "it instead of starting fresh. On by default." + LEAVE,
    "--slot-prompt-similarity": "With several slots: how similar a new prompt must be to reuse "
                                "a slot's memory. This box runs one slot, so it does nothing.",
    "--lora-init-without-apply": "Loads LoRAs switched off, so an app can turn them on per "
                                 "request.",
    "--log-prompts-dir": "Saves every prompt to files in this folder. Fills the disk and "
                         "stores private conversations; debugging only.",
    "--spec-default": "Turns on llama.cpp's standard speculative-decoding recipe. This setup "
                      "already uses the built-in MTP head, which is faster.",
}

# ---------------------------------------------------------------------------
# sd-server options that are not in the curated form
# ---------------------------------------------------------------------------
SD = {
    "--serve-html-path": "Replaces sd-server's own web page with your own HTML file." + LEAVE,
    "--color": "Coloured log output. Only matters in a terminal.",
    "--clip_g": "Second text encoder file, needed by SDXL and SD3 models (not Qwen-Image).",
    "--clip_vision": "Image encoder file, needed for image-prompt features like IP-Adapter and "
                     "Wan image-to-video.",
    "--tokenizer": "A separate tokenizer file, only for the few models that ask for one "
                   "(PiD, Lens).",
    "--high-noise-diffusion-model": "Wan 2.2's video models come in two halves: one for the "
                                    "early, rough steps (high noise) and one for the detail. "
                                    "This is the first half.",
    "--uncond-diffusion-model": "An extra model file used only by Ideogram 4." + LEAVE,
    "--embeddings-connectors": "Extra file needed by LTX audio-video models.",
    "--vae-format": "Forces how the VAE's output is read. 'auto' works for every normal "
                    "model." + LEAVE,
    "--audio-vae": "Sound decoder for LTX models that make video with audio.",
    "--audio-encoder": "Speech encoder for Wan 2.2 S2V, which animates a face to an audio "
                       "track.",
    "--taesd": "A tiny, very fast decoder that turns the result into an image at lower "
               "quality. Good for quick previews on a weak card.",
    "--tae": "Same as the tiny fast decoder (taesd).",
    "--control-net": "A ControlNet model: makes the image follow a sketch, pose or edge map. "
                     "Mostly for SD 1.5/SDXL.",
    "--ip-adapter": "IP-Adapter model: lets you use a picture as part of the prompt. Needs "
                    "the image encoder (clip_vision) too.",
    "--motion-module": "AnimateDiff add-on that turns an SD 1.5 model into a short-video "
                       "model.",
    "--embd-dir": "Folder of 'textual inversion' embeddings (trained words) for SD 1.5/SDXL.",
    "--lora-model-dir": "Folder where LoRA add-on files are kept, so prompts can load them by "
                        "name.",
    "--hires-upscalers-dir": "Folder of upscaler models used by the high-res fix.",
    "--tensor-type-rules": "Re-compresses parts of the model while loading, e.g. to save "
                           "memory. Needs knowledge of the model's internals." + LEAVE,
    "--model-args": "Model-specific switches, e.g. for Chroma or Qwen-Image. Only when a model "
                    "card says so.",
    "--photo-maker": "PhotoMaker model: keeps a person's face consistent across images "
                     "(SDXL).",
    "--pulid-weights": "PuLID model: puts a specific person's face into Flux images.",
    "--upscale-model": "An ESRGAN model that enlarges finished images (e.g. 4x).",
    "--params-backend": "Where weights are kept when not in use: disk, RAM (cpu), or GPU. "
                        "The 'Offload weights to RAM' switch in the form covers the usual "
                        "case." + LEAVE,
    "--split-mode": "With the diffusion model spread over two GPUs: split by whole layers "
                    "(safe default) or by rows (CUDA only)." + LEAVE,
    "--rpc-servers": "Uses other computers on the network as extra GPUs. Slow and "
                     "unencrypted; experiments only.",
    "--disable-prefetch": "Stops loading the next part of the model while the current part "
                          "runs. Slower; only for debugging.",
    "--disable-segmented-compute": "Runs the whole model in one piece even when memory is "
                                   "tight. More likely to run out of memory." + LEAVE,
    "--eager-load": "Loads all weights at start-up instead of when first needed. Slower start, "
                    "faster first image, more RAM held.",
    "--force-sdxl-vae-conv-scale": "Fixes washed-out colours with some SDXL VAEs. Only if "
                                   "your SDXL images look wrong.",
    "--fa": "Faster, lower-memory attention for every part (text encoder too), not just the "
            "diffusion model. Usually safe; turn off if images come out black.",
    "--sage-attn": "An even faster attention method, but CUDA only. Does nothing with the "
                   "Vulkan build.",
    "--diffusion-conv-direct": "A different way of running image convolutions in the "
                               "diffusion model. Sometimes faster on Vulkan, sometimes "
                               "slower - try it and time it.",
    "--vae-conv-direct": "Same, for the VAE decoder. Can reduce memory use when decoding "
                         "large images.",
    "--linear-scale": "Fix for black or noisy images on some cards: scales numbers down to "
                      "avoid overflows. Only if images come out black.",
    "--attn-scale": "Same idea as linear scale, for attention. Only if images come out black "
                    "with flash attention on.",
    "--auto-fit": "on (default): works out by itself what fits on the GPU and puts the rest "
                  "in RAM. Leave it on.",
    "--type": "Converts the model to this precision while loading (e.g. q8_0). Only to save "
              "memory with an uncompressed model file.",
    "--rng": "Which random-number method makes the starting noise. Changes which image a "
             "given seed gives, to match other programs (cuda = A1111, cpu = ComfyUI).",
    "--sampler-rng": "Random-number method used during sampling. Same idea as rng." + LEAVE,
    "--prediction": "Tells the sampler what kind of output the model predicts. Only for odd "
                    "fine-tunes whose card says so.",
    "--lora-apply-mode": "How LoRAs are mixed in: baked in at load (faster) or on the fly "
                         "(safer with compressed models). 'auto' picks correctly.",
    "--ad-model": "ADetailer: a face/hand detector that automatically redraws those areas "
                  "at higher detail.",
    "--ad-prompt": "Prompt used when ADetailer redraws faces/hands. Empty = main prompt.",
    "--ad-negative-prompt": "Negative prompt for the ADetailer redraw.",
    "--extra-ad-args": "Fine-tuning for ADetailer (detection confidence, mask size...)."
                       + LEAVE,
    "--hires-upscaler": "High-res fix: which method enlarges the first image before the "
                        "second detailing pass.",
    "--extra-sample-args": "Advanced sampler tuning. Only when following a model's recipe."
                           + LEAVE,
    "--extra-tiling-args": "Advanced tiling for video decoders." + LEAVE,
    "--ref-image-args": "How reference images for editing are prepared. Empty = automatic, "
                        "which is right for normal use.",
    "--image-preprocess": "How input images are resized or cropped before use." + LEAVE,
    "--high-noise-steps": "Wan 2.2 video: steps for the rough first stage. -1 = automatic.",
    "--clip-skip": "Ignores the last layer(s) of the text encoder. Some anime SD 1.5 models "
                   "want 2. Qwen-Image ignores it.",
    "--batch-count": "How many images each request makes by default.",
    "--qwen-image-layers": "For Qwen-Image 'Layered': how many separate transparent layers to "
                           "split the picture into.",
    "--video-frames": "Video models: how many frames to make. More frames = longer clip, "
                      "much more time and memory.",
    "--fps": "Video: frames per second of the output clip.",
    "--timestep-shift": "Only for NitroFusion models (250 or 500 recommended).",
    "--upscale-repeats": "How many times to run the upscaler (each run enlarges again).",
    "--upscale-tile-size": "Upscaler works on tiles this big. Smaller = less memory, slower.",
    "--hires-width": "High-res fix: final width. 0 = use the scale instead.",
    "--hires-height": "High-res fix: final height. 0 = use the scale instead.",
    "--hires-steps": "High-res fix: steps in the detailing pass. 0 = same as normal steps.",
    "--hires-upscale-tile-size": "High-res fix: tile size for model-based upscalers." + LEAVE,
    "--img-cfg-scale": "Image editing: how closely to stick to the INPUT image (higher = "
                       "change less).",
    "--guidance": "For Flux-style models: how strongly to follow the prompt (their version of "
                  "CFG). 3.5 is the usual value.",
    "--slg-scale": "Skip-layer guidance: can improve anatomy/structure on SD3.5 (2.5 is "
                   "suggested there). 0 = off.",
    "--skip-layer-start": "When skip-layer guidance starts, as a fraction of the steps." + LEAVE,
    "--skip-layer-end": "When skip-layer guidance stops, as a fraction of the steps." + LEAVE,
    "--eta": "How much fresh randomness some samplers add each step. The default for each "
             "sampler is right." + LEAVE,
    "--strength": "Image-to-image: how much to change the input image. 0 = keep it, 1 = "
                  "ignore it completely.",
    "--pm-style-strength": "PhotoMaker: how strongly the style is applied." + LEAVE,
    "--pulid-id-weight": "PuLID: how strongly the person's face is applied.",
    "--control-strength": "ControlNet: how strictly the image follows the sketch/pose (0-1).",
    "--ip-adapter-strength": "IP-Adapter: how strongly the reference picture influences the "
                             "result.",
    "--moe-boundary": "Wan 2.2: at which point generation switches from the rough model to "
                      "the detail model." + LEAVE,
    "--vace-strength": "Wan VACE video editing: how strongly the control video is followed.",
    "--vae-tile-overlap": "With VAE tiling: how much neighbouring tiles overlap. More overlap "
                          "hides seams, costs time.",
    "--hires-scale": "High-res fix: how much to enlarge (2 = double size).",
    "--hires-denoising-strength": "High-res fix: how much the detailing pass may change "
                                  "(0.7 = quite a lot).",
    "--increase-ref-index": "When editing with several reference images, number them 1, 2, "
                            "3... so the prompt can refer to 'image 2'.",
    "--circular": "Makes the image tile seamlessly in both directions (good for textures and "
                  "wallpapers).",
    "--circularx": "Seamless tiling left-right only (panoramas).",
    "--circulary": "Seamless tiling top-bottom only.",
    "--disable-image-metadata": "Don't store the prompt and settings inside the PNG. Turn on "
                                "if you share images and want the prompt private.",
    "--temporal-tiling": "Video: decode the clip in time chunks to save memory. Turn on if "
                         "video decoding runs out of memory.",
    "--hires": "High-res fix: make the image small first, then enlarge and add detail. Gives "
               "big images with fewer mistakes, but takes longer.",
    "--high-noise-sampling-method": "Wan 2.2: sampler for the rough first stage." + LEAVE,
    "--sigmas": "Hand-written noise schedule. Overrides the scheduler; only when copying a "
                "specific recipe.",
    "--hires-sigmas": "Hand-written noise schedule for the high-res pass." + LEAVE,
    "--skip-layers": "Which layers skip-layer guidance skips." + LEAVE,
    "--cache-mode": "Speeds up generation by reusing work between steps (e.g. easycache for "
                    "Qwen-Image/Flux). Often 1.5-2x faster with a small quality cost. Empty = "
                    "off.",
    "--cache-option": "Fine-tuning for the chosen cache mode." + LEAVE,
    "--scm-mask": "Advanced: which steps the cache may skip." + LEAVE,
    "--scm-policy": "Advanced: how the cache decides to skip." + LEAVE,
    "--vae-tile-size": "With VAE tiling: tile size. Smaller = less memory, more seams/time."
                       + LEAVE,
    "--vae-relative-tile-size": "With VAE tiling: tile size relative to the image." + LEAVE,
}
# per-request inputs: set by the app for each job, never at server start
SD_PER_REQUEST = {"--prompt", "-p", "--prompt-file", "--negative-prompt-file", "--init-img", "-i",
                  "--end-img", "--mask", "--control-image", "--ip-adapter-image", "--control-video",
                  "--pm-id-images-dir", "--pm-id-embed-path", "--pulid-id-embedding", "--ref-image",
                  "-r", "--ref-video", "--ref-video-audio", "--ref-audio", "--audio"}
# sd-server options that take no value (its --help does not mark them)
SD_SWITCHES = {"--color", "--disable-prefetch", "--disable-segmented-compute", "--eager-load",
               "--force-sdxl-vae-conv-scale", "--fa", "--sage-attn", "--diffusion-conv-direct",
               "--vae-conv-direct", "--increase-ref-index", "--circular", "--circularx",
               "--circulary", "--disable-image-metadata", "--temporal-tiling", "--hires"}

# ---------------------------------------------------------------------------
# plain words for the curated llama.cpp fields whose tips were too technical
# ---------------------------------------------------------------------------
PARAM_PLAIN = {
    "REASONING_BUDGET": "The most words the model may spend 'thinking' before it must answer. "
                        "-1 = no limit. Lower it if answers take too long to start.",
    "PORT": "The network port apps connect to for this model.",
    "MODEL": "Which model file this instance runs.",
    "YARN_EXT_FACTOR": "YaRN stretches a model to a longer context than it was trained on. "
                       "This is one of its fine-tuning dials. Leave at -1 (automatic).",
    "YARN_ATTN_FACTOR": "YaRN fine-tuning dial. Leave at -1 (automatic).",
    "YARN_BETA_FAST": "YaRN fine-tuning dial. Leave at -1 (automatic).",
    "YARN_BETA_SLOW": "YaRN fine-tuning dial. Leave at -1 (automatic).",
    "SPEC_NGRAM_MOD_N_MIN": "For the 'ngram' guess-ahead modes, which predict the next words by "
                            "spotting repeats of earlier text. Only used if speculative type is "
                            "ngram-mod. Leave at the default.",
    "SPEC_NGRAM_SIMPLE_SIZE_N": "ngram guess-ahead: how many recent words to match. Only for "
                                "ngram-simple. Leave at the default.",
    "SPEC_NGRAM_SIMPLE_SIZE_M": "ngram guess-ahead: how many words to guess at once. Only for "
                                "ngram-simple.",
    "SPEC_NGRAM_SIMPLE_MIN_HITS": "ngram guess-ahead: how often a pattern must appear before "
                                  "it is trusted. Only for ngram-simple.",
    "SPEC_NGRAM_MAP_K_SIZE_N": "Same as the ngram lookup size, for ngram-map-k.",
    "SPEC_NGRAM_MAP_K_SIZE_M": "Same as the ngram guess size, for ngram-map-k.",
    "SPEC_NGRAM_MAP_K_MIN_HITS": "Same as the ngram minimum hits, for ngram-map-k.",
    "SPEC_NGRAM_MAP_K4V_SIZE_N": "Same as the ngram lookup size, for ngram-map-k4v.",
    "SPEC_NGRAM_MAP_K4V_SIZE_M": "Same as the ngram guess size, for ngram-map-k4v.",
    "SPEC_NGRAM_MAP_K4V_MIN_HITS": "Same as the ngram minimum hits, for ngram-map-k4v.",
    "TYPICAL_P": "A word-picking rule that skips words that are 'too surprising'. 1.0 = off. "
                 "Most people leave it off.",
    "REPEAT_LAST_N": "How far back (in tokens) the repetition penalty looks for repeats.",
    "FREQUENCY_PENALTY": "Makes words that were already used a lot less likely. 0 = off. "
                         "Small values (0.1-0.5) reduce repetitive wording.",
    "DRY_BASE": "Part of the DRY anti-repetition rule: how fast the penalty grows for longer "
                "repeats. Only used when DRY strength is above 0.",
    "DRY_PENALTY_LAST_N": "How far back the DRY rule looks for repeats. 0 = off.",
    "XTC_THRESHOLD": "XTC removes the most obvious word choices to make writing more creative. "
                     "1.0 = off. Bad for code; only for creative writing.",
    "DYNATEMP_RANGE": "Lets 'temperature' (randomness) go up and down automatically depending "
                      "on how sure the model is. 0 = off.",
    "DYNATEMP_EXP": "How strongly the automatic temperature reacts. Only used when the range "
                    "above is not 0.",
    "MIROSTAT_LR": "Mirostat keeps the text's 'surprise level' steady. This is how quickly it "
                   "adjusts. Only used when Mirostat is on.",
    "MIROSTAT_ENT": "The surprise level Mirostat aims for (higher = more varied text). Only "
                    "used when Mirostat is on.",
    "GGML_VK_PERF_LOGGER_CONCURRENT": "Debugging: logs GPU timings for work that runs at the "
                                      "same time. Only with the perf logger on.",
    "GGML_VK_PERF_LOGGER_FREQUENCY": "Debugging: how often GPU timings are logged. Only with "
                                     "the perf logger on.",
    "MAIN_GPU": "With several GPUs: which one holds the shared parts. Only matters with split "
                "mode none or row.",
}


# ---------------------------------------------------------------------------
# generated explanations for options nobody has written text for yet
# ---------------------------------------------------------------------------
def plain(flag, desc, engine):
    table = SD if engine == "sd.cpp" else LLAMA
    if flag in table:
        return table[flag]
    d = desc or ""
    base = flag
    for pre, what in (("--high-noise-", "Wan 2.2 video models run in two stages; this is the "
                                        "same as the option without 'high-noise', but only for "
                                        "the rough first stage. Irrelevant for images."),
                      ("--spec-draft-", "Settings for a separate small draft model used to "
                                        "guess ahead. This setup uses the built-in MTP head "
                                        "instead, so it usually does nothing."),
                      ("--cors-", "Rules for which web pages in a browser may call this server."),
                      ("--log-", "Changes how the log is written. The panel reads the log, so "
                                 "keep it readable."),
                      ("--models-", "Router mode (one server swapping models on demand), which "
                                    "the panel's instances replace."),
                      ("--hires-", "High-res fix (make small, enlarge, add detail)."),
                      ("--vae-", "The VAE turns the finished result into a picture."),
                      ("--lora", "LoRAs are small add-on files that change a model's style or "
                                 "skills.")):
        if base.startswith(pre):
            return what + " Upstream: " + d + LEAVE
    if base.endswith("-batch"):
        return "Same as the option without '-batch', but only for reading the prompt." + LEAVE
    if "experimental" in d.lower():
        return "Experimental upstream feature: " + d + LEAVE
    if re.search(r"\bpath\b|\bfile\b|FNAME|<string>", d + flag):
        return "Points at a file or folder: " + d + LEAVE
    return "Upstream describes it as: " + d + LEAVE
