#!/usr/bin/env python3
"""
stable-diffusion.cpp instances: parameters, launch plan, and generation
(added 2026-09-23).

An instance whose instance.json says engine="sd.cpp" runs sd-server instead of
llama-server. Everything instance-shaped stays shared with llama.cpp: the
systemd --user unit, devices, ports, the failed-start counter, the launch log,
start/stop. What differs lives here:

  * its own parameter set (SD_* keys in the instance's params.env)
  * presets that name the files a model family needs and where to get them
  * launch_plan(): argv + env for sd-server, device pinning, file/RAM checks
  * generation: jobs go to sd-server's async API (/sdcpp/v1/img_gen), the
    panel polls them and keeps the images in ~/sdcpp/outputs/<instance>/

Placement: the first device of the instance runs the diffusion model and the
VAE. The text encoder goes to the CPU, the same card, or the instance's second
device (SD_TE_ON). --offload-to-cpu keeps weights in host RAM and moves each
module to the GPU only while it runs, which is what lets a 7B diffusion model
plus an 8B text encoder work on a 6 GB card.
"""
import base64, json, os, re, shlex, subprocess, threading, time, urllib.error, urllib.request, uuid
from pathlib import Path

P = None
E = None                          # engines module

DEFAULTS = dict(
    BACKEND="vulkan", PORT=8084, HOST="127.0.0.1", SD_PRESET="",
    SD_DIFFUSION_MODEL="", SD_MODEL="", SD_LLM="", SD_LLM_VISION="", SD_VAE="",
    SD_CLIP_L="", SD_T5XXL="",
    SD_TE_ON="cpu", SD_VAE_ON="gpu", SD_OFFLOAD=1, SD_DIFFUSION_FA=1, SD_VAE_TILING=0,
    SD_MMAP=1, SD_MAX_VRAM="", SD_THREADS=-1,
    SD_WIDTH=1024, SD_HEIGHT=1024, SD_STEPS=20, SD_CFG=6.0, SD_SAMPLER="euler",
    SD_SCHEDULER="", SD_FLOW_SHIFT="", SD_SEED=-1, SD_NEGATIVE="",
    SD_LOG_LEVEL="info", SD_EXTRA="", RAM_FLOOR_MB=2048,
)
FILE_KEYS = ("SD_DIFFUSION_MODEL", "SD_MODEL", "SD_LLM", "SD_LLM_VISION", "SD_VAE",
             "SD_CLIP_L", "SD_T5XXL")
BACKENDS = ("vulkan", "rocm", "cpu")
SAMPLERS = ["euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2", "ipndm",
            "ipndm_v", "lcm", "ddim_trailing", "tcd", "res_multistep", "res_2s", "er_sde",
            "euler_cfg_pp", "euler_a_cfg_pp", "euler_ge", "dpm++2m_sde", "dpm++2m_sde_bt", "lms"]
SCHEDULERS = ["", "discrete", "karras", "exponential", "ays", "gits", "sgm_uniform", "simple",
              "smoothstep", "kl_optimal", "lcm", "bong_tangent", "logit_normal", "flux2", "flux",
              "beta"]
MODEL_EXT = (".gguf", ".safetensors", ".sft", ".ckpt")

# --------------------------------------------------------------------------
# presets: which files a model family needs, and where they come from
# --------------------------------------------------------------------------
HF = "https://huggingface.co"
PRESETS = {
    "qwen-image-2.1": dict(
        label="Qwen-Image 2.1 (text-to-image)",
        notes=("7B diffusion model with Qwen3-VL-8B as its text encoder and its own VAE "
               "(the Qwen-Image 1 and Wan VAEs do not work with it). Use sizes divisible "
               "by 32. Upstream example: cfg 6.0, euler."),
        files=dict(
            SD_DIFFUSION_MODEL=dict(match=r"qwen[-_]image[-_]2[._]1.*\.gguf$",
                                    url=f"{HF}/leejet/Qwen-Image-2.1-GGUF/resolve/main/qwen_image_2.1-Q4_K.gguf",
                                    size_gb=4.4),
            SD_LLM=dict(match=r"^qwen3-?vl-8b-instruct-(?!.*mmproj).*\.gguf$",
                        url=f"{HF}/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
                        size_gb=5.0),
            SD_VAE=dict(match=r"qwen_image_2\.1_vae.*\.safetensors$",
                        url=f"{HF}/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors",
                        size_gb=0.7),
        ),
        optional=dict(
            SD_LLM_VISION=dict(match=r"mmproj-qwen3-?vl-8b-instruct.*\.gguf$",
                               url=f"{HF}/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf",
                               size_gb=0.75, why="needed only for image editing"),
        ),
        values=dict(SD_CFG=6.0, SD_SAMPLER="euler", SD_STEPS=20, SD_WIDTH=1024, SD_HEIGHT=1024,
                    SD_DIFFUSION_FA=1, SD_OFFLOAD=1, SD_SCHEDULER="", SD_FLOW_SHIFT=""),
        small_card=dict(SD_TE_ON="cpu", SD_VAE_TILING=1),
        big_card=dict(SD_TE_ON="gpu", SD_VAE_TILING=0),
    ),
}

# --------------------------------------------------------------------------
# parameter metadata, in the shape the Parameters tab already renders
# --------------------------------------------------------------------------
GROUPS = ["Backend", "Model files", "Placement & memory", "Generation defaults", "Server"]


def _m(group, label, tip, type="text", options=None, strict=False, unit=None, numeric=False):
    d = dict(group=group, label=label, tip=tip, type=type)
    if options is not None:
        d["options"] = options
    if strict:
        d["strict"] = True
    if unit:
        d["unit"] = unit
    if numeric:
        d["numeric"] = True
    return d


def meta():
    files = [f["path"] for f in model_files()]
    fo = [""] + files
    M = {
        "BACKEND": _m("Backend", "Compute backend",
                      "Which stable-diffusion.cpp build runs this instance. <b>vulkan</b> drives "
                      "both the 7900 XTX (RADV) and the RTX 2060 (NVIDIA's Vulkan driver); upstream "
                      "ships no Linux CUDA build. <b>rocm</b> is AMD only. <b>cpu</b> is slow "
                      "(minutes per step for a 7B model) but needs no GPU. The build itself is "
                      "chosen on the Builds tab.", "select", list(BACKENDS), strict=True),
        "SD_PRESET": _m("Model files", "Preset",
                        "Informational: the preset last applied. Apply presets from the card above "
                        "the form; it fills the file paths and the recommended settings.",
                        "select", [""] + list(PRESETS)),
        "SD_DIFFUSION_MODEL": _m("Model files", "Diffusion model",
                                 "--diffusion-model. The standalone diffusion transformer/UNet "
                                 "(Qwen-Image, Flux, Wan, ...). Leave empty when using a full "
                                 "checkpoint below.", "select", fo),
        "SD_MODEL": _m("Model files", "Full checkpoint",
                       "-m. A single-file checkpoint that bundles everything (SD 1.5, SDXL). Leave "
                       "empty for split models.", "select", fo),
        "SD_LLM": _m("Model files", "Text encoder (LLM)",
                     "--llm. The language-model text encoder: Qwen3-VL-8B for Qwen-Image 2.1, "
                     "Qwen2.5-VL-7B for Qwen-Image 1, Mistral for Flux 2.", "select", fo),
        "SD_LLM_VISION": _m("Model files", "Text encoder vision (mmproj)",
                            "--llm_vision. The vision projector for the text encoder. Only needed "
                            "for image editing with a GGUF text encoder.", "select", fo),
        "SD_VAE": _m("Model files", "VAE", "--vae. Must match the model family.", "select", fo),
        "SD_CLIP_L": _m("Model files", "CLIP-L", "--clip_l (Flux 1, SD3).", "select", fo),
        "SD_T5XXL": _m("Model files", "T5-XXL", "--t5xxl (Flux 1, SD3).", "select", fo),
        "SD_TE_ON": _m("Placement & memory", "Text encoder on",
                       "<b>cpu</b>: keeps the 5 GB encoder off a small card; it runs once per "
                       "prompt, so the cost is seconds. <b>gpu</b>: the instance's first card. "
                       "<b>second</b>: the instance's second card, when it has one.",
                       "select", ["cpu", "gpu", "second"], strict=True),
        "SD_VAE_ON": _m("Placement & memory", "VAE on",
                        "Where the VAE decodes. <b>gpu</b> is much faster; use <b>cpu</b> only if "
                        "the decode runs out of VRAM even with tiling.",
                        "select", ["gpu", "cpu"], strict=True),
        "SD_OFFLOAD": _m("Placement & memory", "Offload weights to RAM",
                         "--offload-to-cpu. Weights live in host RAM and each module is moved to "
                         "the GPU only while it runs. Needed on a 6 GB card; costs some speed.",
                         "bool"),
        "SD_DIFFUSION_FA": _m("Placement & memory", "Flash attention (diffusion)",
                              "--diffusion-fa. Less VRAM for attention, usually faster.", "bool"),
        "SD_VAE_TILING": _m("Placement & memory", "VAE tiling",
                            "--vae-tiling. Decodes in tiles so a 1024px image fits a small card.",
                            "bool"),
        "SD_MMAP": _m("Placement & memory", "Memory-map files", "--mmap.", "bool"),
        "SD_MAX_VRAM": _m("Placement & memory", "VRAM budget",
                          "--max-vram, in GiB, e.g. 5 or vulkan0=5. Empty lets it use what is free.",
                          unit="GiB"),
        "SD_THREADS": _m("Placement & memory", "CPU threads", "-t. -1 uses the physical cores.",
                         "int", numeric=True),
        "RAM_FLOOR_MB": _m("Placement & memory", "Host RAM floor",
                           "The launcher kills sd-server if MemAvailable drops below this. The "
                           "host has hard-locked from running out of RAM.", "int", unit="MB",
                           numeric=True),
        "SD_WIDTH": _m("Generation defaults", "Width", "-W. Default for requests that do not set it.",
                       "int", unit="px", numeric=True),
        "SD_HEIGHT": _m("Generation defaults", "Height", "-H.", "int", unit="px", numeric=True),
        "SD_STEPS": _m("Generation defaults", "Steps", "--steps.", "int", numeric=True),
        "SD_CFG": _m("Generation defaults", "CFG scale", "--cfg-scale.", "float", numeric=True),
        "SD_SAMPLER": _m("Generation defaults", "Sampler", "--sampling-method.", "select", SAMPLERS),
        "SD_SCHEDULER": _m("Generation defaults", "Scheduler",
                           "--scheduler. Empty = the model's own default.", "select", SCHEDULERS),
        "SD_FLOW_SHIFT": _m("Generation defaults", "Flow shift",
                            "--flow-shift. Empty = automatic (resolution-dependent for Qwen-Image 2.1).",
                            numeric=True),
        "SD_SEED": _m("Generation defaults", "Seed", "-s. Negative = random.", "int", numeric=True),
        "SD_NEGATIVE": _m("Generation defaults", "Negative prompt", "-n. Default negative prompt."),
        "PORT": _m("Server", "Port", "--listen-port.", "int", numeric=True),
        "HOST": _m("Server", "Listen address",
                   "--listen-ip. sd-server has no authentication: keep 127.0.0.1 and use it "
                   "through the panel.", "select", ["127.0.0.1", "0.0.0.0"]),
        "SD_LOG_LEVEL": _m("Server", "Log level", "--log-level.", "select",
                           ["info", "verbose", "debug", "warn", "error"], strict=True),
        "SD_EXTRA": _m("Server", "Extra arguments",
                       "Anything else sd-server accepts, passed as-is (shell-quoted)."),
    }
    for k, tip in PLAIN_TIPS.items():
        if k in M:
            M[k]["tip"] = tip + f"<br><br><span style='opacity:.75'>Flag: {_FLAG.get(k, k)}</span>"
    return M


_FLAG = dict(SD_DIFFUSION_MODEL="--diffusion-model", SD_MODEL="-m", SD_LLM="--llm",
             SD_LLM_VISION="--llm_vision", SD_VAE="--vae", SD_CLIP_L="--clip_l",
             SD_T5XXL="--t5xxl", SD_OFFLOAD="--offload-to-cpu", SD_DIFFUSION_FA="--diffusion-fa",
             SD_VAE_TILING="--vae-tiling", SD_MMAP="--mmap", SD_MAX_VRAM="--max-vram",
             SD_THREADS="-t", SD_WIDTH="-W", SD_HEIGHT="-H", SD_STEPS="--steps",
             SD_CFG="--cfg-scale", SD_SAMPLER="--sampling-method", SD_SCHEDULER="--scheduler",
             SD_FLOW_SHIFT="--flow-shift", SD_SEED="-s", SD_NEGATIVE="-n", PORT="--listen-port",
             HOST="--listen-ip", SD_LOG_LEVEL="--log-level", SD_TE_ON="--backend te=...",
             SD_VAE_ON="--backend vae=...", BACKEND="(which build)")
PLAIN_TIPS = dict(
    BACKEND="Which kind of program runs the model. <b>vulkan</b> works on both graphics cards "
            "(the 7900 XTX and the RTX 2060) - pick this. <b>rocm</b> is AMD-only. <b>cpu</b> "
            "uses no graphics card and is very slow (minutes per image).",
    SD_PRESET="The model recipe last applied. Use the Model preset card above to fill "
              "everything in for you.",
    SD_DIFFUSION_MODEL="The main image model - the part that actually draws. For Qwen-Image "
                       "2.1 this is the qwen-image-2.1 .gguf file.",
    SD_MODEL="Only for older all-in-one model files (Stable Diffusion 1.5 / SDXL). Leave "
             "empty for Qwen-Image.",
    SD_LLM="The text encoder: a language model that reads your prompt and explains it to the "
           "image model. Qwen-Image 2.1 needs Qwen3-VL-8B.",
    SD_LLM_VISION="Lets the text encoder look at pictures. Only needed for editing an existing "
                  "image; leave empty to save memory.",
    SD_VAE="Turns the model's internal result into an actual picture. Must be the one made "
           "for this model - a wrong VAE gives garbage colours.",
    SD_CLIP_L="Extra text encoder for Flux 1 and SD3 models. Not used by Qwen-Image.",
    SD_T5XXL="Another extra text encoder for Flux 1 and SD3. Not used by Qwen-Image.",
    SD_TE_ON="Where the text encoder runs. <b>cpu</b>: saves graphics memory, costs a few "
             "seconds per image - best on the 6 GB RTX 2060. <b>gpu</b>: fastest, needs "
             "memory. <b>second</b>: on this instance's second graphics card.",
    SD_VAE_ON="Where the picture is decoded at the end. <b>gpu</b> is much faster. Pick "
              "<b>cpu</b> only if the last step runs out of graphics memory.",
    SD_OFFLOAD="Keeps the models in normal RAM and moves each part onto the graphics card only "
               "while it works. Needed on small cards; a bit slower. Leave on.",
    SD_DIFFUSION_FA="A faster, less memory-hungry way of doing the model's maths. Leave on. "
                    "Turn off only if pictures come out black.",
    SD_VAE_TILING="Decodes the final picture in pieces so big images fit on a small card. "
                  "Turn on for the 6 GB card or if the last step runs out of memory.",
    SD_MMAP="Reads model files straight from disk as needed instead of copying them into "
            "memory first. Faster start, less RAM. Leave on.",
    SD_MAX_VRAM="Caps how much graphics memory it may use, in GB (e.g. 5). Empty = use "
                "whatever is free. Set it if something else shares the card.",
    SD_THREADS="How many CPU cores to use for work done on the CPU. -1 = automatic (all "
               "real cores).",
    RAM_FLOOR_MB="Safety net: if free RAM drops below this, the server is stopped before the "
                 "whole machine freezes. Do not set it lower than 2048.",
    SD_WIDTH="Picture width in pixels when an app doesn't say. Must be a multiple of 32. "
             "Bigger = slower and more memory.",
    SD_HEIGHT="Picture height in pixels when an app doesn't say. Must be a multiple of 32.",
    SD_STEPS="How many times the model refines the picture. More steps = more detail but "
             "slower (time grows in a straight line). 20-30 is typical.",
    SD_CFG="How strictly the picture follows your prompt. Too low = ignores you, too high = "
           "harsh, over-cooked images. Qwen-Image 2.1 recommends about 6.",
    SD_SAMPLER="The method used to refine the picture step by step. <b>euler</b> is the safe "
               "choice for Qwen-Image; others can look different but rarely better.",
    SD_SCHEDULER="How the refining is spread over the steps. Empty = the model's own choice, "
                 "which is almost always right.",
    SD_FLOW_SHIFT="Shifts effort between the rough shape and the fine detail. Empty = "
                  "automatic, which is right for Qwen-Image 2.1.",
    SD_SEED="The random starting point. The same seed + same settings = the same picture. "
            "-1 = a new random picture each time.",
    SD_NEGATIVE="Things you do NOT want in the picture, e.g. 'blurry, text, watermark'. Used "
                "when an app sends none.",
    PORT="The network port the image server listens on. The panel talks to it for you.",
    HOST="Who may connect. <b>127.0.0.1</b> = only this machine (safe; use the panel's "
         "Generate card). <b>0.0.0.0</b> = anyone on the network, with NO password.",
    SD_LOG_LEVEL="How chatty the log is. <b>info</b> is right; <b>debug</b> only to chase a "
                 "problem.",
    SD_EXTRA="Any other option, typed by hand. Easier: tick them in 'All other options from "
             "this build' below.",
)


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------
def model_files():
    """Candidate weight files: ~/models/sd/ and ~/models/ (not recursive beyond sd/)."""
    out = []
    for d in (P.MODELS / "sd", P.MODELS):
        try:
            for f in sorted(d.iterdir()):
                if f.is_file() and f.name.lower().endswith(MODEL_EXT) and not f.name.startswith("."):
                    out.append(dict(path=str(f), name=f.name, dir=d.name,
                                    size_gb=round(f.stat().st_size / 1e9, 2)))
        except OSError:
            pass
    return out


def _partial(path):
    """A file still being downloaded by the panel or by curl."""
    p = Path(path)
    return (p.parent / f".{p.name}.curl.log").exists() and any(
        d.get("dest") == str(p) and d.get("status") == "running" for d in P._downloads.values())


def preset_status(name, values=None):
    """For a preset: which role each local file would fill, and what is missing."""
    pr = PRESETS.get(name)
    if not pr:
        raise ValueError(f"unknown preset {name!r}")
    files = model_files()
    values = values or {}
    roles = []
    for opt, table in ((False, pr["files"]), (True, pr.get("optional", {}))):
        for key, spec in table.items():
            rx = re.compile(spec["match"], re.I)
            cur = values.get(key)
            if cur and rx.search(Path(cur).name) and Path(cur).exists():
                pick = cur
            else:
                hits = [f for f in files if rx.search(f["name"])]
                pick = hits[0]["path"] if hits else None
            dl = any(d.get("dest") == str(P.MODELS / "sd" / spec["url"].rsplit("/", 1)[-1])
                     and d.get("status") in ("running", "unpacking")
                     for d in P._downloads.values())
            roles.append(dict(key=key, optional=opt, path=pick, found=bool(pick),
                              downloading=dl, url=spec["url"], size_gb=spec["size_gb"],
                              file=spec["url"].rsplit("/", 1)[-1], why=spec.get("why")))
    return dict(name=name, label=pr["label"], notes=pr["notes"], roles=roles,
                ready=all(r["found"] and not r["downloading"] for r in roles if not r["optional"]))


def apply_preset(name, values, card_vram_mib=None):
    st = preset_status(name, values)
    pr = PRESETS[name]
    new = dict(values)
    for r in st["roles"]:
        # optional roles (e.g. the vision projector, for editing only) stay as they
        # are: loading them costs RAM on every start
        if r["found"] and not r["optional"]:
            new[r["key"]] = r["path"]
    new.update(pr["values"])
    small = card_vram_mib is not None and card_vram_mib < 12000
    new.update(pr["small_card" if (small or card_vram_mib is None) else "big_card"])
    new["SD_PRESET"] = name
    return new, st


def download_missing(name, include_optional=False):
    st = preset_status(name)
    (P.MODELS / "sd").mkdir(exist_ok=True)
    started = []
    for r in st["roles"]:
        if r["found"] or r["downloading"] or (r["optional"] and not include_optional):
            continue
        started.append(P.start_download(r["url"], r["file"], subdir="sd"))
    return dict(started=started, status=preset_status(name))


# --------------------------------------------------------------------------
# params
# --------------------------------------------------------------------------
def _coerce(vals):
    for k, v in list(vals.items()):
        d = DEFAULTS.get(k)
        if isinstance(d, bool):
            continue
        if isinstance(d, int) and str(v).lstrip("-").isdigit():
            vals[k] = int(v)
        elif isinstance(d, float):
            try:
                vals[k] = float(v)
            except (TypeError, ValueError):
                pass
    return vals


def load_params(inst):
    cur = dict(DEFAULTS)
    cur.update(P._read_env_file(inst["dir"] / "params.env"))
    return _coerce(cur)


def write_params(inst, vals, note=""):
    body = ["# GENERATED by the admin panel - do not hand-edit.",
            f"# stable-diffusion.cpp instance '{inst['id']}'. Read by panel/instance_launch.py.",
            *([f"# {note}"] if note else []), ""]
    for k in DEFAULTS:
        v = vals.get(k, DEFAULTS[k])
        body.append(f"{k}={P._env_quote(v)}")
    (inst["dir"] / "params.env").write_text("\n".join(body) + "\n")


def save_params(inst, new):
    cur = load_params(inst)
    unknown = sorted(set(new) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"not stable-diffusion.cpp settings: {', '.join(unknown)}")
    cur.update(new)
    cur = _coerce(cur)
    if cur["BACKEND"] not in BACKENDS:
        raise ValueError(f"BACKEND must be one of {', '.join(BACKENDS)}")
    try:
        port = int(cur["PORT"])
        assert 1024 <= port <= 65535 and port not in (8090, 8091, 8092)
    except (ValueError, AssertionError):
        raise ValueError("PORT must be 1024-65535 and not the panel's own ports")
    for k in ("SD_WIDTH", "SD_HEIGHT"):
        if int(cur[k]) % 32:
            raise ValueError(f"{k} must be divisible by 32")
    try:
        shlex.split(str(cur.get("SD_EXTRA") or ""))
    except ValueError as e:
        raise ValueError(f"SD_EXTRA does not parse: {e}")
    write_params(inst, cur)
    return cur


# --------------------------------------------------------------------------
# launch plan
# --------------------------------------------------------------------------
def _vk_index(bindir, icds, dev):
    """Vulkan index of `dev` as THIS sd.cpp build enumerates with these ICDs."""
    joined = ":".join(icds)
    out = P.sh(f"env -u GGML_VK_VISIBLE_DEVICES VK_DRIVER_FILES={shlex.quote(joined)} "
               f"VK_ICD_FILENAMES={shlex.quote(joined)} LD_LIBRARY_PATH={shlex.quote(bindir)} "
               f"timeout 60 {shlex.quote(bindir)}/sd-server --list-devices 2>/dev/null", timeout=70)
    for m in re.finditer(r"^Vulkan(\d+)\t(.*)$", out, re.M):
        if P._vk_matches(dev, m.group(2)):
            return int(m.group(1))
    return None


def _free_vram_mib(dev):
    if dev.get("driver") == "amdgpu":
        base = f"/sys/bus/pci/devices/{dev['pci']}"
        tot = P.read_int(f"{base}/mem_info_vram_total") // 1048576
        used = P.read_int(f"{base}/mem_info_vram_used") // 1048576
        return tot - used if tot else None
    if dev.get("driver") == "nvidia":
        r = P._nvidia_smi(dev["pci"]) or {}
        try:
            return int(float(r.get("mem_total_mib"))) - int(float(r.get("mem_used_mib") or 0))
        except (TypeError, ValueError):
            return None
    return None


def launch_plan(inst, values=None):
    v = dict(values or load_params(inst))
    errors, warnings = [], []
    backend = str(v.get("BACKEND") or "vulkan")
    bindir = E.ENGINES["sd.cpp"].active(backend)
    if not bindir:
        errors.append(f"no active stable-diffusion.cpp {backend} build - install one and press "
                      f"'Use for {backend}' on the Builds tab")
    alld = {d["pci"]: d for d in P.gpu_devices(probe=False)}
    devs, missing = [], []
    for pci in inst["devices"]:
        if pci == "cpu":
            continue
        d = alld.get(pci)
        if not d:
            missing.append(pci)
        else:
            devs.append(d)
    for pci in missing:
        errors.append(f"device {pci} is not in this machine")
    if backend == "cpu" and devs:
        warnings.append("BACKEND=cpu: the selected GPU is not used at all")
    if backend != "cpu" and not devs and not missing:
        errors.append(f"BACKEND={backend} needs a GPU; this instance is CPU-only - pick BACKEND=cpu")

    # files
    if not (v.get("SD_DIFFUSION_MODEL") or v.get("SD_MODEL")):
        errors.append("no model: set a diffusion model or a full checkpoint (or apply a preset)")
    for k in FILE_KEYS:
        f = v.get(k)
        if f and not Path(f).is_file():
            errors.append(f"{k}: {f} does not exist")
        elif f and _partial(f):
            errors.append(f"{k}: {Path(f).name} is still downloading")

    env = dict(GGML_VK_ALLOW_SYSMEM_FALLBACK="0")
    placement = []
    gpu_name = None
    if backend == "vulkan" and devs:
        prim = devs[0]
        venv, _m, verr = P.vulkan_pinning(devs, None, check_runtime=False)
        errors += [e for e in verr if "enumerates" not in e]
        env.update({k: val for k, val in venv.items() if k.startswith(("VK_", "GGML_"))})
        env.pop("GGML_VK_VISIBLE_DEVICES", None)
        idx = {}
        if bindir and not errors:
            icds = venv["VK_DRIVER_FILES"].split(":")
            for d in devs[:2]:
                i = _vk_index(bindir, icds, d)
                if i is None:
                    errors.append(f"this build cannot see {d['name']} through Vulkan "
                                  f"(driver {d.get('driver')})")
                idx[d["pci"]] = i
        gpu_name = f"vulkan{idx.get(prim['pci'], 0)}"
        te = {"cpu": "cpu", "gpu": gpu_name}.get(v.get("SD_TE_ON"))
        if v.get("SD_TE_ON") == "second":
            if len(devs) < 2:
                errors.append("SD_TE_ON=second but the instance has only one device")
            else:
                te = f"vulkan{idx.get(devs[1]['pci'], 1)}"
        vae = "cpu" if v.get("SD_VAE_ON") == "cpu" else gpu_name
        placement = [f"diffusion={gpu_name}", f"te={te}", f"vae={vae}"]
    elif backend == "rocm" and devs:
        if len(devs) > 1 or devs[0].get("driver") != "amdgpu":
            errors.append("BACKEND=rocm drives exactly one AMD card")
        env.update(HIP_VISIBLE_DEVICES="0", ROCR_VISIBLE_DEVICES="0")
        gpu_name = "rocm0"
        te = "cpu" if v.get("SD_TE_ON") == "cpu" else gpu_name
        placement = [f"diffusion={gpu_name}", f"te={te}",
                     f"vae={'cpu' if v.get('SD_VAE_ON') == 'cpu' else gpu_name}"]
    else:
        placement = ["cpu"]

    # memory: with offload every weight sits in host RAM; without it, on the GPU
    sizes = {k: Path(v[k]).stat().st_size for k in FILE_KEYS
             if v.get(k) and Path(v[k]).is_file()}
    total_mib = sum(sizes.values()) // 1048576
    avail = P._meminfo_mb("MemAvailable")
    host_mib = total_mib if (_truthy(v.get("SD_OFFLOAD")) or backend == "cpu") else \
        sizes.get("SD_LLM", 0) // 1048576 if v.get("SD_TE_ON") == "cpu" else 0
    if avail and host_mib and host_mib + int(v.get("RAM_FLOOR_MB") or 2048) > avail:
        errors.append(f"needs ~{host_mib} MiB of host RAM for weights; only {avail} MiB is "
                      f"available with the {v.get('RAM_FLOOR_MB')} MiB floor - this box has "
                      "hard-locked from running out of RAM")
    elif avail and host_mib and host_mib > avail * 0.7:
        warnings.append(f"weights take ~{host_mib} MiB of the {avail} MiB host RAM available")
    if devs and backend != "cpu":
        free = _free_vram_mib(devs[0])
        on_gpu = total_mib if not _truthy(v.get("SD_OFFLOAD")) else \
            max([s // 1048576 for k, s in sizes.items()
                 if not (k in ("SD_LLM", "SD_LLM_VISION") and v.get("SD_TE_ON") == "cpu")] or [0])
        if free is not None and on_gpu and on_gpu + 1024 > free:
            other = [s.get("instance") or f"pid {s['pid']}" for s in P.list_servers()
                     if any(g.get("pdev") == devs[0]["pci"] for g in s.get("gpu") or [])]
            warnings.append(f"{P._short_gpu(devs[0]['name'])} has ~{free} MiB free"
                            + (f" (in use by {', '.join(map(str, other))})" if other else "")
                            + f"; this needs ~{on_gpu + 1024} MiB at its peak. With "
                            "GGML_VK_ALLOW_SYSMEM_FALLBACK=0 it will fail to allocate rather "
                            "than run slowly.")

    port = int(v.get("PORT") or 0)
    for other in P.instance_ids():
        if other == inst["id"]:
            continue
        try:
            o = P.get_instance(other)
        except ValueError:
            continue
        if str(P._read_env_file(o["dir"] / "params.env").get("PORT") or
               (P.DEFAULTS["PORT"] if o["legacy"] else "")) == str(port):
            warnings.append(f"PORT {port} is shared with instance '{other}'")

    argv = [f"{bindir or '<no build>'}/sd-server", "--listen-ip", str(v.get("HOST") or "127.0.0.1"),
            "--listen-port", str(port)]
    for k, flag in (("SD_DIFFUSION_MODEL", "--diffusion-model"), ("SD_MODEL", "-m"),
                    ("SD_LLM", "--llm"), ("SD_LLM_VISION", "--llm_vision"), ("SD_VAE", "--vae"),
                    ("SD_CLIP_L", "--clip_l"), ("SD_T5XXL", "--t5xxl")):
        if v.get(k):
            argv += [flag, str(v[k])]
    if placement and placement != ["cpu"]:
        argv += ["--backend", ",".join(placement)]
    elif backend == "cpu" or not devs:
        argv += ["--backend", "cpu"]
    for k, flag in (("SD_OFFLOAD", "--offload-to-cpu"), ("SD_DIFFUSION_FA", "--diffusion-fa"),
                    ("SD_VAE_TILING", "--vae-tiling"), ("SD_MMAP", "--mmap")):
        if _truthy(v.get(k)):
            argv.append(flag)
    if str(v.get("SD_MAX_VRAM") or "").strip():
        argv += ["--max-vram", str(v["SD_MAX_VRAM"]).strip()]
    argv += ["-t", str(v.get("SD_THREADS", -1)),
             "-W", str(v["SD_WIDTH"]), "-H", str(v["SD_HEIGHT"]), "--steps", str(v["SD_STEPS"]),
             "--cfg-scale", str(v["SD_CFG"]), "--sampling-method", str(v["SD_SAMPLER"]),
             "-s", str(v["SD_SEED"]), "--log-level", str(v.get("SD_LOG_LEVEL") or "info")]
    if v.get("SD_SCHEDULER"):
        argv += ["--scheduler", str(v["SD_SCHEDULER"])]
    if str(v.get("SD_FLOW_SHIFT") or "").strip():
        argv += ["--flow-shift", str(v["SD_FLOW_SHIFT"])]
    if v.get("SD_NEGATIVE"):
        argv += ["-n", str(v["SD_NEGATIVE"])]
    try:
        argv += shlex.split(str(v.get("SD_EXTRA") or ""))
    except ValueError as e:
        errors.append(f"SD_EXTRA does not parse: {e}")
    if str(v.get("HOST")) in ("0.0.0.0", "::"):
        warnings.append("sd-server has no authentication and is bound to every interface")
    env["LD_LIBRARY_PATH"] = bindir or ""
    return dict(engine="sd.cpp", argv=argv, env=env, errors=errors, warnings=warnings,
                backend=backend, bindir=bindir, rundir=str(inst["rundir"]),
                device=dict(name=devs[0]["name"] if devs else "CPU", pci=devs[0]["pci"] if devs else "cpu"),
                placement=placement, weights_mib=total_mib, host_ram_mib=host_mib,
                vulkan_map=[], template_copy=None, api_key=None)


def _truthy(x):
    return str(x).strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------
# the running server
# --------------------------------------------------------------------------
def describe(pid, argv):
    """/api/servers row for an sd-server process."""
    g = lambda *names: P._argv_get(argv, names)
    host = g("--listen-ip") or "127.0.0.1"
    port = g("--listen-port") or "1234"
    model = g("--diffusion-model") or g("-m", "--model") or ""
    exe = os.path.realpath(f"/proc/{pid}/exe") if os.path.exists(f"/proc/{pid}/exe") else argv[0]
    caps = _http(host, port, "/sdcpp/v1/capabilities", timeout=1.5)
    health = "ok" if isinstance(caps, dict) and not caps.get("_error") else \
        ("no answer" if caps is None else "loading")
    return dict(engine="sd.cpp", model=model, model_name=Path(model).name if model else None,
                binary=exe, host=host, port=int(port) if str(port).isdigit() else port,
                backend=next((b for b in ("rocm", "vulkan", "cpu") if b in exe.lower()), "cpu"),
                health=health, ctx=None, kv=None, ngl=None, parallel=None, spec=None,
                mmproj=g("--llm_vision"), cache_ram=None, draft=None, alias=None,
                batch=None, ubatch=None, device=g("--backend"), log_file=None, slots=None,
                slots_busy=None, live_tps=None, visible_devices={})


def _http(host, port, path, body=None, timeout=10, method=None):
    h = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    req = urllib.request.Request(f"http://{h}:{port}{path}",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 method=method or ("POST" if body is not None else "GET"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read() or b"{}")
        except ValueError:
            detail = {}
        return dict(_error=e.code, detail=detail)
    except Exception:
        return None


def _server_addr(inst):
    with P.using_instance(inst):
        pid = P.server_pid()
    if not pid:
        raise ValueError(f"{inst['id']} is not running - press Start")
    argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
    return (P._argv_get(argv, ("--listen-ip",)) or "127.0.0.1",
            P._argv_get(argv, ("--listen-port",)) or "1234")


def health(inst):
    try:
        host, port = _server_addr(inst)
    except ValueError:
        return None
    caps = _http(host, port, "/sdcpp/v1/capabilities", timeout=2)
    if caps is None:
        return dict(status="loading")
    if caps.get("_error"):
        return dict(status="error", code=caps["_error"])
    return dict(status="ok")


# --------------------------------------------------------------------------
# generation jobs
# --------------------------------------------------------------------------
_jobs = {}                       # panel job id -> record
_jlock = threading.Lock()


def _outdir(iid):
    d = P.HOME / "sdcpp" / "outputs" / iid
    d.mkdir(parents=True, exist_ok=True)
    return d


def generate(inst, body):
    host, port = _server_addr(inst)
    v = load_params(inst)
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("empty prompt")
    w = int(body.get("width") or v["SD_WIDTH"])
    h = int(body.get("height") or v["SD_HEIGHT"])
    if w % 32 or h % 32 or not (64 <= w <= 4096 and 64 <= h <= 4096):
        raise ValueError("width and height must be 64-4096 and divisible by 32")
    steps = int(body.get("steps") or v["SD_STEPS"])
    cfg = float(body.get("cfg") if body.get("cfg") not in (None, "") else v["SD_CFG"])
    seed = int(body.get("seed") if body.get("seed") not in (None, "") else v["SD_SEED"])
    req = dict(prompt=prompt, negative_prompt=str(body.get("negative") or v.get("SD_NEGATIVE") or ""),
               width=w, height=h, seed=seed, batch_count=1, output_format="png",
               sample_params=dict(sample_method=str(body.get("sampler") or v["SD_SAMPLER"]),
                                  sample_steps=steps, guidance=dict(txt_cfg=cfg)))
    if v.get("SD_SCHEDULER"):
        req["sample_params"]["scheduler"] = v["SD_SCHEDULER"]
    r = _http(host, port, "/sdcpp/v1/img_gen", req, timeout=30)
    if r is None:
        raise ValueError("sd-server did not answer")
    if r.get("_error"):
        raise ValueError(f"sd-server refused the job (HTTP {r['_error']}): "
                         f"{(r.get('detail') or {}).get('error') or r.get('detail')}")
    jid = uuid.uuid4().hex[:12]
    rec = dict(id=jid, instance=inst["id"], server_job=r.get("id"), status=r.get("status"),
               submitted=time.time(), started=None, finished=None, request=req,
               files=[], error=None, host=host, port=port)
    with _jlock:
        _jobs[jid] = rec
        for old in sorted(_jobs, key=lambda k: _jobs[k]["submitted"])[:-50]:
            _jobs.pop(old, None)
    threading.Thread(target=_poll, args=(rec,), daemon=True).start()
    return _public(rec)


def _poll(rec):
    while True:
        r = _http(rec["host"], rec["port"], f"/sdcpp/v1/jobs/{rec['server_job']}", timeout=15)
        if r is None:
            rec.update(status="failed", error="lost contact with sd-server", finished=time.time())
            return
        if r.get("_error"):
            rec.update(status="failed", error=f"job lookup HTTP {r['_error']}", finished=time.time())
            return
        rec["status"] = r.get("status")
        rec["queue_position"] = r.get("queue_position")
        if r.get("started") and not rec["started"]:
            rec["started"] = time.time()
        if r.get("status") == "completed":
            out = _outdir(rec["instance"])
            stamp = time.strftime("%Y%m%d_%H%M%S")
            for img in (r.get("result") or {}).get("images", []):
                fn = f"{stamp}_{rec['id']}_{img.get('index', 0)}.png"
                (out / fn).write_bytes(base64.b64decode(img["b64_json"]))
                rec["files"].append(fn)
            meta = dict(rec["request"], files=rec["files"], seconds=round(time.time() - rec["submitted"], 1),
                        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            (out / f"{stamp}_{rec['id']}.json").write_text(json.dumps(meta, indent=1))
            rec["finished"] = time.time()
            return
        if r.get("status") in ("failed", "cancelled"):
            rec.update(error=(r.get("error") or {}).get("message") or r.get("status"),
                       finished=time.time())
            return
        time.sleep(1.5)


def _public(rec):
    r = {k: v for k, v in rec.items() if k not in ("host", "port")}
    end = rec["finished"] or time.time()
    r["elapsed_s"] = round(end - rec["submitted"], 1)
    return r


def job(jid):
    with _jlock:
        rec = _jobs.get(jid)
    if not rec:
        raise ValueError("no such job")
    return _public(rec)


def cancel(jid):
    with _jlock:
        rec = _jobs.get(jid)
    if not rec:
        raise ValueError("no such job")
    r = _http(rec["host"], rec["port"], f"/sdcpp/v1/jobs/{rec['server_job']}/cancel", {}, timeout=10)
    return dict(ok=bool(r) and not r.get("_error"), detail=r)


def gallery(iid, limit=24):
    d = _outdir(iid)
    out = []
    for m in sorted(d.glob("*.json"), reverse=True)[:limit]:
        try:
            meta = json.loads(m.read_text())
        except (OSError, ValueError):
            continue
        out.append(dict(files=meta.get("files", []), prompt=meta.get("prompt"),
                        width=meta.get("width"), height=meta.get("height"),
                        seed=meta.get("seed"), seconds=meta.get("seconds"),
                        steps=(meta.get("sample_params") or {}).get("sample_steps"),
                        created=meta.get("created")))
    active = [_public(j) for j in _jobs.values() if j["instance"] == iid and not j["finished"]]
    return dict(items=out, active=active)


def image_path(iid, name):
    if not re.match(r"^[\w.-]+\.png$", name or ""):
        raise ValueError("bad file name")
    f = _outdir(iid) / name
    if not f.is_file():
        raise ValueError("no such image")
    return f


def bind(panel_module, engines_module):
    global P, E
    P, E = panel_module, engines_module
