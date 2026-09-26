# Changelog

## 1.0.1 — 2026-09-26

Everything since 1.0.0.

### Serving many requests at once
- **Gateway admission queue.** A server is never sent more requests than it has slots; the rest
  wait at the gateway, interactive before batch, first-come within each.
- **KV-aware admission for llama.cpp.** When a server's slots share one KV pool (`--kv-unified`),
  a request goes only when the tokens in flight plus its own fit the pool with the ~1.7x
  headroom llama.cpp needs. Measured on an RTX 2060 (8192-token shared pool, 16 slots, 48
  requests): straight to llama-server **0 of 48** succeeded ("Context size has been exceeded");
  through the gateway **48 of 48**, 235 t/s.
- **Slot planning** in the memory calculator: how many conversations run safely at the
  instance's typical depth, the CTX all slots need, and the most the card could hold with
  separate or shared KV. PARALLEL now offers 16 and 32.
- **`/v1/batches`**: OpenAI Batch API shape with inline input, run at batch priority, JSONL
  results, owner-only, resumes after a panel restart, deleted 7 days after finishing.
- **vLLM engine** (`vllm_engine.py`, `install-vllm.sh`): vLLM as a LexiPanel instance, with
  settings, launch plan, card guard and health. The API key never appears on the command line,
  in the plan or in logs. The gateway sends vLLM the model name it serves.

### Measuring and changing safely
- **Bench tab** (`benchlab.py`, `benchstats.py`): profile, calibrate, A/B-B/A compare, goodput
  and traffic analysis. Every number comes with an interval and a verdict (better / worse /
  small / same / undecided), plus an A/A self-test of the false-win rate. It never restarts a
  server that is serving a real request.
- **Auto-fit verification with intervals**; a bias in the optimizer's consistency score fixed;
  a 6-hour cool-down after a change that had to be rolled back.
- **Safe restarts** (`restarts.py`): journaled, drained (the gateway holds requests instead of
  failing them), verified against the intended settings and a canary output, rolled back to the
  last known-good settings when verification fails **or the server never comes back**, and
  reported. They refuse to run under another measurement and recover after a panel crash.

### Many machines
- **Remote endpoints**: servers on other machines, registered by address, appear in the server
  list and the gateway. Link-local, cloud-metadata, multicast and reserved addresses are
  refused; keys are stored 0600.
- **Fleet v2**: from the primary, start, stop, safely restart, or change the model and sizing of
  a member's instances, and drain a box out of the gateway. Commands ride back in the replies to
  the member's own reports (members open no port). Each is signed with a per-box key, runs
  once, expires in 10 minutes, is re-checked against the member's own allowlist (nothing
  allowed by default), and is audited on both boxes. No shell commands, no file transfer, no
  port, key or free-form argument changes. A member's planned restart makes the primary's
  gateway re-route or hold that instance's requests: in a live run, 210 requests went through a
  member's restart and all 210 succeeded.

### Fixes
- The panel runs as whoever installed it; no hard-coded user or home folder in `panel.py`.
- A partial settings update (only CTX, say) keeps the instance's backend instead of switching it
  to Vulkan's settings file.
- `make-backup.sh` backs up the panel it belongs to.
- The power helper reports its watchdog off when no watchdog device exists.
- Bench goodput uses the same depth floor for llama.cpp and vLLM.

### Repository
- `.gitignore` and the CI workflow (`.github/workflows/ci.yml`) are now in the repository;
  CI no longer needs a particular home folder. `UPLOAD-FIRST.txt` removed.

### Upgrading from 1.0.0
- Update the code, then restart the panel (`systemctl --user restart inf01-panel`, or your
  `LexiPanel-panel` unit). Settings, instances and results are kept.
- Fleet across machines through Caddy: add `/api/fleet/event` to the paths that skip the login
  (see `systemd/Caddyfile.new`); the panel checks the box token itself.
- vLLM instances need `bash install-vllm.sh` once.

## 1.0.0 — 2026-09-25

First public release.
