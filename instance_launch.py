#!/usr/bin/env python3
"""
LexiPanel instance launcher:  instance_launch.py <id> [--dry-run]

Runs one non-main inference instance (panel/instances/<id>/). Started by the
user unit LexiPanel-inst@<id>.service, or directly by the panel when there is no
systemd --user bus.

Everything about WHAT runs comes from panel.launch_plan(), the same function
the panel shows before you press Start - there is no second copy of the flag
logic to drift. This file only owns the process lifecycle the bash scripts
carry for main:

  * fallback tiers     consecutive failed starts pick normal -> safe -> minimal
  * RAM-floor watchdog kills llama-server before host RAM runs out (the box has
                       hard-locked from exactly that)
  * start counter      cleared once a start stays up LAUNCH_OK_SECONDS
  * log archive        engine log copied to ~/llama_logs/engine_debug_inst-<id>_*
                       on exit, because llama-server truncates it on every start

Exit 78 means the plan refused (bad config); the unit does not retry that.
main is deliberately refused: it belongs to LexiPanel-llama and its scripts.
"""
import hashlib, json, os, shutil, signal, subprocess, sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel as P                                   # noqa: E402

LAUNCH_OK_SECONDS = 120
EX_CONFIG = 78


def say(out, msg):
    line = f"[LexiPanel-inst] {msg}"
    print(line, flush=True)
    if out:
        out.write((line + "\n").encode())
        out.flush()


def in_instance(inst, fn):
    """Thread target bound to the instance. The panel resolves paths through a
    THREAD-LOCAL current instance, so a bare thread falls back to main - a test
    run of this file wrote its counter reset into main's .launch-fails that way."""
    def run():
        with P.using_instance(inst):
            fn()
    return run


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: instance_launch.py <id> [--dry-run]")
    iid, dry = sys.argv[1], "--dry-run" in sys.argv[2:]
    inst = P.get_instance(iid)
    if inst["legacy"]:
        sys.exit("main is launched by LexiPanel-llama and run_llama_*.sh, not by this file")
    if inst.get("engine") in P.GEN_ENGINES:
        return run_sd(inst, dry)

    with P.using_instance(inst):
        P.LOGS.mkdir(exist_ok=True)
        out = None if dry else open(inst["launch_out"], "wb")

        fail_file = inst["dir"] / ".launch-fails"
        fails = P.fail_count()
        tier = "normal" if fails == 0 else ("safe" if fails == 1 else "minimal")
        values = P.load_params()
        if tier != "normal":
            tv = P.read_tier(tier)
            if tv:
                say(out, f"{fails} consecutive failed starts - using the '{tier}' tier")
                values = tv
            else:
                say(out, f"wanted tier '{tier}' but it has not been saved - keeping params")
        if not dry:
            fail_file.write_text(f"{fails + 1}\n")

        plan = P.launch_plan(values)
        say(out, f"instance {iid} | device {plan['device'].get('name')} ({inst['device']}) | "
                 f"backend {plan['backend']} | build {plan['bindir']}")
        for w in plan["warnings"]:
            say(out, f"WARN  {w}")
        for e in plan["errors"]:
            say(out, f"ABORT {e}")
        env_show = " ".join(f"{k}={v}" for k, v in sorted(plan["env"].items()))
        say(out, f"env  {env_show}")
        say(out, "argv " + " ".join(plan["argv"]))
        if plan["errors"]:
            if not dry:
                # a refused config is not a crashed start: do not walk the tiers
                fail_file.write_text(f"{fails}\n")
            sys.exit(EX_CONFIG)
        if dry:
            return

        rundir = Path(plan["rundir"])
        (rundir / "slots").mkdir(parents=True, exist_ok=True)
        (rundir / "telemetry").mkdir(parents=True, exist_ok=True)
        if plan["template_copy"]:
            shutil.copyfile(plan["template_src"], plan["template_copy"])
            got = hashlib.sha256(Path(plan["template_copy"]).read_bytes()).hexdigest()
            if got != str(values.get("TEMPLATE_SHA256")):
                say(out, f"ABORT template sha256 mismatch after copy: {got}")
                sys.exit(EX_CONFIG)
        if plan["api_key"]:
            kf = rundir / "api-keys"
            fd = os.open(kf, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(plan["api_key"] + "\n")

        env = dict(os.environ)
        for k in [k for k in env if k.startswith(("GGML_", "VK_", "CUDA_", "HIP_", "ROCR_", "HSA_"))]:
            env.pop(k)                  # nothing inherited may override the plan
        env.update(plan["env"])
        env["LD_LIBRARY_PATH"] = plan["env"]["LD_LIBRARY_PATH"] + (
            ":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")

        if plan["backend"] == "vulkan":
            # Exactly one device may enumerate. Anything else means the ICD pin
            # did not isolate the card this instance is assigned to.
            r = subprocess.run([plan["argv"][0], "--list-devices"], env=env,
                               capture_output=True, text=True, timeout=90)
            devs = [l.strip() for l in (r.stdout + r.stderr).splitlines()
                    if l.strip().startswith("Vulkan")]
            say(out, "vulkan devices: " + (" | ".join(devs) or "(none)"))
            want = len(inst.get("devices") or [inst["device"]])
            if len(devs) != want:
                say(out, f"ABORT expected exactly {want} Vulkan device(s), got {len(devs)}")
                sys.exit(EX_CONFIG)
            # the plan resolved VulkanN names from a cached enumeration; re-check
            # them against this fresh one, since -dev/-ot/--spec-draft-device use them
            for m in plan.get("vulkan_map") or []:
                line = next((d for d in devs if d.startswith(m["name"] + ":")), "")
                if not P._vk_matches(P._device_record(m["pci"]) or {}, line):
                    say(out, f"ABORT {m['name']} is now '{line}', expected {m['device']}")
                    sys.exit(EX_CONFIG)

        child = subprocess.Popen(plan["argv"], env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, cwd=str(P.LLAMA))
        say(out, f"llama-server pid {child.pid}")

        def forward(sig, _frm):
            say(out, f"signal {sig} - stopping llama-server")
            try:
                child.send_signal(signal.SIGTERM)
            except OSError:
                pass
        signal.signal(signal.SIGTERM, forward)
        signal.signal(signal.SIGINT, forward)

        def pump():
            for line in iter(child.stdout.readline, b""):
                sys.stdout.buffer.write(line)
                sys.stdout.flush()
                out.write(line)
                out.flush()
        threading.Thread(target=in_instance(inst, pump), daemon=True).start()

        floor = int(str(values.get("RAM_FLOOR_MB") or 0) or 0)

        def watchdog():
            t0 = time.time()
            cleared = False
            while child.poll() is None:
                avail = P._meminfo_mb("MemAvailable:")
                if floor and avail and avail < floor:
                    say(out, f"[WATCHDOG] MemAvailable {avail}MB < {floor}MB - killing "
                             "llama-server before the host locks up")
                    child.kill()
                    return
                if not cleared and time.time() - t0 >= LAUNCH_OK_SECONDS:
                    fail_file.write_text("0\n")
                    say(out, f"up {LAUNCH_OK_SECONDS}s on tier '{tier}' - failure counter reset")
                    cleared = True
                time.sleep(2)
        threading.Thread(target=in_instance(inst, watchdog), daemon=True).start()

        def banner():
            log = rundir / "telemetry/engine_debug.log"
            for _ in range(180):
                if child.poll() is not None:
                    return
                if log.exists() and log.stat().st_size:
                    break
                time.sleep(1)
            meta = (f"### LexiPanel-engine BACKEND={plan['backend']} INSTANCE={iid} "
                    f"DEVICE={inst['device']} CTX={values.get('CTX')} KV={values.get('KV_TYPE')} "
                    f"B={values.get('BATCH')} UB={values.get('UBATCH')} "
                    f"SPEC={values.get('SPEC_TYPE') or 'none'} TIER={tier} "
                    f"STARTED={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
            try:
                with open(log, "a") as f:
                    f.write(meta + "\n")
                Path(str(log) + ".meta").write_text(meta + "\n")
            except OSError:
                pass
        threading.Thread(target=in_instance(inst, banner), daemon=True).start()

        rc = child.wait()
        log = rundir / "telemetry/engine_debug.log"
        if log.exists():
            ts = time.strftime("%Y%m%d_%H%M%S")
            dst = P.LOGS / f"engine_debug_inst-{iid}_{plan['backend']}_{ts}.log"
            try:
                shutil.copyfile(log, dst)
                if Path(str(log) + ".meta").exists():
                    shutil.copyfile(str(log) + ".meta", str(dst) + ".meta")
                say(out, f"log archived to {dst}")
            except OSError as e:
                say(out, f"log archive failed: {e}")
        say(out, f"llama-server exited {rc}")
        sys.exit(0 if rc in (0, -signal.SIGTERM, 143) else (rc if rc > 0 else 1))


def run_sd(inst, dry):
    """sd-server or audiocpp_server for a stable-diffusion.cpp / audio.cpp
    instance. Same contract as the llama path: failed-start counter, RAM-floor
    watchdog, launch log, log archive, exit 78 for a refused plan."""
    iid = inst["id"]
    engine = inst["engine"]
    srv = {"sd.cpp": "sd-server", "audio.cpp": "audiocpp_server"}.get(engine, engine)
    tag = engine.replace(".", "")
    with P.using_instance(inst):
        P.LOGS.mkdir(exist_ok=True)
        out = None if dry else open(inst["launch_out"], "wb")
        fail_file = inst["dir"] / ".launch-fails"
        fails = P.fail_count()
        values = P.load_params()
        if not dry:
            fail_file.write_text(f"{fails + 1}\n")
        plan = P.launch_plan(values)
        say(out, f"instance {iid} | engine {engine} | device {plan['device'].get('name')} | "
                 f"backend {plan['backend']} | build {plan['bindir']} | "
                 f"placement {','.join(plan['placement'])}")
        for w in plan["warnings"]:
            say(out, f"WARN  {w}")
        for e in plan["errors"]:
            say(out, f"ABORT {e}")
        say(out, "env  " + " ".join(f"{k}={v}" for k, v in sorted(plan["env"].items())))
        say(out, "argv " + " ".join(plan["argv"]))
        if plan.get("config") is not None:
            say(out, "config " + json.dumps(plan["config"]))
        if plan["errors"]:
            if not dry:
                fail_file.write_text(f"{fails}\n")
            sys.exit(EX_CONFIG)
        if dry:
            return
        rundir = Path(plan["rundir"])
        (rundir / "telemetry").mkdir(parents=True, exist_ok=True)
        if plan.get("config_file"):
            Path(plan["config_file"]).write_text(json.dumps(plan["config"], indent=1) + "\n")
        log = rundir / "telemetry/engine_debug.log"
        env = dict(os.environ)
        for k in [k for k in env if k.startswith(("GGML_", "VK_", "CUDA_", "HIP_", "ROCR_", "HSA_"))]:
            env.pop(k)
        env.update(plan["env"])
        mod = P.GEN_ENGINES.get(engine)             # secrets (e.g. VLLM_API_KEY): to the process only -
        if hasattr(mod, "secrets"):                  # never in the plan, the API or the log above
            env.update(mod.secrets(inst))
        child = subprocess.Popen(plan["argv"], env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, cwd=str(rundir))
        say(out, f"{srv} pid {child.pid}")

        def forward(sig, _frm):
            say(out, f"signal {sig} - stopping {srv}")
            try:
                child.send_signal(signal.SIGTERM)
            except OSError:
                pass
        signal.signal(signal.SIGTERM, forward)
        signal.signal(signal.SIGINT, forward)
        logf = open(log, "wb")
        logf.write((f"### LexiPanel-engine ENGINE={engine} BACKEND={plan['backend']} INSTANCE={iid} "
                    f"STARTED={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n").encode())

        def pump():
            for line in iter(child.stdout.readline, b""):
                sys.stdout.buffer.write(line)
                sys.stdout.flush()
                for f in (out, logf):
                    f.write(line)
                    f.flush()
        threading.Thread(target=pump, daemon=True).start()
        floor = int(str(values.get("RAM_FLOOR_MB") or 0) or 0)

        def watchdog():
            t0, cleared = time.time(), False
            while child.poll() is None:
                avail = P._meminfo_mb("MemAvailable:")
                if floor and avail and avail < floor:
                    say(out, f"[WATCHDOG] MemAvailable {avail}MB < {floor}MB - killing "
                             f"{srv} before the host locks up")
                    child.kill()
                    return
                if not cleared and time.time() - t0 >= LAUNCH_OK_SECONDS:
                    fail_file.write_text("0\n")
                    say(out, f"up {LAUNCH_OK_SECONDS}s - failure counter reset")
                    cleared = True
                time.sleep(2)
        threading.Thread(target=watchdog, daemon=True).start()
        rc = child.wait()
        logf.close()
        ts = time.strftime("%Y%m%d_%H%M%S")
        try:
            shutil.copyfile(log, P.LOGS / f"engine_debug_inst-{iid}_{tag}_{ts}.log")
        except OSError as e:
            say(out, f"log archive failed: {e}")
        say(out, f"{srv} exited {rc}")
        sys.exit(0 if rc in (0, -signal.SIGTERM, 143) else (rc if rc > 0 else 1))


if __name__ == "__main__":
    main()
