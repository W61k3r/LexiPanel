# LexiPanel HTTP API

The panel's UI is a single page (`static/index.html`) over this JSON API. The panel binds `127.0.0.1:8090` and is meant to sit behind Caddy with basic auth — never expose it directly.

**Request rules (cross-site protection).** Browsers resend the panel's basic-auth login to
requests from any website, so the API refuses anything that looks like it came from another
site: a `Sec-Fetch-Site` of `cross-site`/`same-site`, a foreign `Origin`, and POSTs with the
content types browsers can send cross-site without a preflight (`application/x-www-form-urlencoded`,
`multipart/form-data`, `text/plain`) — all with HTTP 403. Scripts are unaffected as long as a
POST with a body sends `Content-Type: application/json` (plain `curl -d` defaults to
form-urlencoded and will be refused).

Almost every call is **per instance**: add `?inst=<id>` (default `main`). POST bodies are JSON. Errors come back as `{"error": "..."}` with HTTP 400/404/500; a call that does not apply to the instance's engine returns `{"na": true, ...}`.

Route list generated from `panel.py`'s handlers.

## Status & servers

| Method | Route |
|---|---|
| GET | `/api/servers` |
| GET | `/api/status` |
| POST | `/api/restart` |
| POST | `/api/start` |
| POST | `/api/stop` |
| POST | `/api/main/reset-failed` |

**main's start guard.** For the legacy `main` instance, `/api/status` includes `guard`: unit state, starts used in
systemd's start-limit window (`starts_in_window` of `burst` per `interval_s`), `locked` / `unlocks_at`, `loading`,
the failure counter and next fallback tier, `can_reset`, the command that clears a lockout, and recent refusals.
`/api/start` and `/api/restart` refuse with `{ok:false, code:"start_limit"|"loading", msg, guard, systemd}`:
`start_limit` while systemd's crash-loop guard is active, `loading` for a restart while main is still loading
(send `{"force": true}` to restart anyway). `/api/main/reset-failed` clears the lockout when the sudo rule allows it.

## Instances

| Method | Route |
|---|---|
| GET | `/api/devices` |
| GET | `/api/gpus` |
| GET | `/api/instance-profiles` |
| GET | `/api/instances` |
| GET | `/api/launch-plan` |
| POST | `/api/instance-profile/apply` |
| POST | `/api/instance-profile/copy` |
| POST | `/api/instance-profile/delete` |
| POST | `/api/instance-profile/rename` |
| POST | `/api/instance-profile/save` |
| POST | `/api/instance/create` |
| POST | `/api/instance/delete` |
| POST | `/api/instance/update` |
| POST | `/api/launch-plan` |

## Parameters, tiers & profiles

| Method | Route |
|---|---|
| GET | `/api/estimate` |
| GET | `/api/flags` |
| GET | `/api/param-meta` |
| GET | `/api/params` |
| GET | `/api/profiles` |
| GET | `/api/ram-budget` |
| GET | `/api/tiers` |
| POST | `/api/estimate` |
| POST | `/api/params` |
| POST | `/api/profile/copy` |
| POST | `/api/profile/delete` |
| POST | `/api/profile/load` |
| POST | `/api/profile/new` |
| POST | `/api/profile/rename` |
| POST | `/api/profile/save` |
| POST | `/api/ram-budget` |
| POST | `/api/tier/copy` |
| POST | `/api/tier/delete` |
| POST | `/api/tier/load` |
| POST | `/api/tier/reset-fails` |
| POST | `/api/tier/save` |

## Optimizer & depth curve

| Method | Route |
|---|---|
| GET | `/api/curve` |
| GET | `/api/optimize/run` |
| GET | `/api/optimize/runs` |
| GET | `/api/optimize/status` |
| GET | `/api/optimize/suite` |
| GET | `/api/speed-curve` |
| POST | `/api/curve/start` |
| POST | `/api/curve/stop` |
| POST | `/api/optimize/apply` |
| POST | `/api/optimize/rollback` |
| POST | `/api/optimize/start` |
| POST | `/api/optimize/stop` |

`/api/optimize/start` also takes `workload` (`{depths, weights, prompt_tokens, output_tokens}`:
speed is then the time a typical request of that mix takes, from decode measured at those
depths), `phases: ["candidates"]` with `candidates: [{label, launch}]` (up to 6 alternatives,
each the baseline plus its own overrides; never `MODEL`, `BACKEND`, `PORT` or `HOST`) and
`source`. Auto-fit uses these; the Optimize tab does not need them.

## Bench (measurements with intervals)

| Method | Route |
|---|---|
| GET | `/api/bench` |
| GET | `/api/bench/calibration` |
| GET | `/api/bench/run` |
| GET | `/api/bench/traffic` |
| POST | `/api/bench/delete` |
| POST | `/api/bench/start` |
| POST | `/api/bench/stop` |

`/api/bench/start` takes `kind`: `profile` (the running server, no restart), `calibrate`
(A/A plus the self-test; `restart: true` restarts between blocks), `compare` (A/B with
`candidate: {"KEY": "value"}`, restarts for every switch) or `goodput` (`concurrency`,
`goodput_depth`, `slo_ttft_s`, `slo_tps`). Common options: `preset` (`workload` | `quick`) or
`depths`, `reps`, `n_predict`, `margin` (0.02 = 2 %), `alpha`, `looks`, `blocks`,
`sampled_reps`, `quiet_s`, `max_wait_s`. Runs that restart the server need
`allow_restart: true`. `candidate` refuses `PORT`, `HOST`, `BACKEND`, `SPEC_DRAFT_MODEL`,
`MMPROJ` and unknown keys, and must fit the estimated VRAM. `/api/bench/traffic?days=14` needs
no GPU. `/api/bench/run?id=` returns a run with every block's raw samples.

## Gateway queue and batch jobs

| Method | Route |
|---|---|
| GET | `/v1/batches` |
| POST | `/v1/batches` |

Every request through the gateway waits for a free slot on the server it is routed to
(llama.cpp parallel slots, vLLM's requests at once; unknown servers are not limited), and
interactive requests always go before batch ones. Mark a request as background work with the
header `X-LexiPanel-Priority: batch` (or a `priority` field, removed before forwarding).
`/api/gateway` reports each server's slots, busy, waiting (batch), served, mean wait and
timeouts, and the batch jobs by status.

`POST /v1/batches` with `{"model": default, "input": [{"custom_id", "body": {chat completion}}],
"metadata": {}}` (or `input_jsonl`) creates a batch; `GET /v1/batches/<id>` shows its status and
`request_counts`, `GET /v1/batches/<id>/output` returns the results as JSONL,
`POST /v1/batches/<id>/cancel` and `/delete` stop and remove it. A user only ever sees their own
batches. Unlike live requests, a batch keeps its prompts and results on disk (0600) until it is
deleted, at most 7 days after it finished. Up to 10,000 requests or 50 MB per batch.

## vLLM instances

| Method | Route |
|---|---|
| GET | `/api/vllm/status` |

An instance with `engine: "vllm"` (create with `/api/instance/create`, one NVIDIA or AMD card)
runs `vllm serve` from its own venv (`bash install-vllm.sh`, `--rocm` for AMD). Its settings are
the `VL_*` keys (see `/api/param-meta` on that instance); the API key is write-only (shown as
`********`). It appears in `/api/servers` and the gateway like any instance, and Bench measures it
(`profile`, `goodput` up to 32 streams). `/api/vllm/status` reports each runtime's vLLM / torch
versions and devices.

## Remote endpoints

| Method | Route |
|---|---|
| GET | `/api/remotes` |
| POST | `/api/remotes/add` |
| POST | `/api/remotes/delete` |
| POST | `/api/remotes/test` |

Model servers on other machines, registered by address so they appear in `/api/servers` and
`/api/status` `servers` (with `remote: {name, url, kind, gateway}`) next to local processes.
`add` (admin): `name`, `url` (`http(s)://host:port` only), optional `api_key`, `kind`
(`auto` | `llama.cpp` | `vllm` | `openai`) and `gateway` (also serve it at `/v1` under its
name and model names). `test` probes an address without saving it or its key. Link-local
(cloud metadata), multicast, unspecified and reserved addresses are refused and re-checked at
every poll; redirects are not followed; keys are stored 0600 and never returned.

## Safe restarts

| Method | Route |
|---|---|
| GET | `/api/restarts` |
| GET | `/api/restarts/export` |
| GET | `/api/restarts/report` |
| POST | `/api/restarts/settings` |
| POST | `/api/restarts/start` |

`/api/restarts` returns the instance's restart in progress, settings, a 30-day summary and
the recent history. `/api/restarts/report?id=` is one restart's full journal (every step with
its time, the intended settings, the verification and recovery results). `/api/restarts/export`
is CSV (`?days=90`). `/api/restarts/start` (operator) runs a safe restart in the background;
the audit log records the authenticated caller, never a name from the body.
`/api/restarts/settings` (admin): `kv_handoff` (experimental, default off), `canary` (default
on), `hold_s` (gateway hold, default 120) and `drain_s` (wait for requests in flight, default 600).

## Workload & auto-fit

llama.cpp instances only (`{"na": true}` otherwise). Every call is per instance (`?inst=<id>`).

| Method | Route | |
|---|---|---|
| GET | `/api/workload` | `?days=7..60` (default 14). `envelope` (requests, depth / output / decode percentiles, concurrency, the 168 hours of the week with busy share and learned idle windows, depth histogram, real vs measured curve, the depth mix, the last configuration change), `findings` (each with evidence and, where one can be measured, a `candidate`), `configs`, and `autofit` (settings, what is due and whether it may start now, the running experiment, proposals, history) |
| POST | `/api/workload/settings` | any of `{mode: off\|propose\|auto, window: "learned"\|"HH-HH", quiet_min, max_per_week, min_gain, max_minutes, reshape, goal}` |
| POST | `/api/workload/experiment` | `{kind: "tune"\|"reshape"}` runs one now (skips the window, quiet and weekly checks, not the others) |
| POST | `/api/workload/stop` | stops the running experiment; the optimizer restores the saved settings |
| POST | `/api/workload/proposal/apply` | `{id, restart}` applies a proposal; it is then verified on real traffic like any change |
| POST | `/api/workload/proposal/dismiss` | `{id}`; that candidate is not measured again for this configuration |
| POST | `/api/workload/rollback` | `{id, restart}` puts back what that experiment's apply replaced (a setting changed by hand since is left alone) |

## Gateway (one endpoint, quotas, usage)

| Method | Route | |
|---|---|---|
| GET | `/v1/models` | every running llama.cpp / ONNX Runtime / Camelid instance, by name |
| POST | `/v1/chat/completions` | OpenAI chat completion routed by `model`; streaming passes through; per-user quotas (429) |
| GET | `/api/gateway` | the model names, quotas and usage per day / user / model |
| POST | `/api/gateway/quota` | admin: `{user, rpm, tokens_day, concurrent, models}` (0 or empty = no limit) |

## Access (users, roles, keys, audit)

Multi-user mode: every route needs a login (HTTP Basic, or `Authorization: Bearer lp_...`) and a role. GETs are viewer (files, debug bundle, backup: admin); POSTs are admin unless listed for operators in `auth.py`.

| Method | Route | |
|---|---|---|
| GET | `/api/auth` | mode and who you are; admins also get users, keys (no secrets) and the audit tail with its chain check |
| POST | `/api/auth/mode` | `{mode: single|multi, admin_user, admin_password}` |
| POST | `/api/auth/users` | `{user, password, role}` add or change |
| POST | `/api/auth/users/delete` | `{user}` (never the last admin) |
| POST | `/api/auth/keys/create` | `{user, role, days, label}` -> the key, shown once |
| POST | `/api/auth/keys/revoke` | `{id}` |

## Fleet

Full LexiPanel on every box, one primary (docs/FLEET.md). View-only across boxes.

| Method | Route | |
|---|---|---|
| GET | `/api/fleet` | this box's role, name, box id; the primary adds every box (state online / stale / offline and its last report); a member adds whether it joined and its last send |
| POST | `/api/fleet/settings` | `{role: standalone|primary|member, name, primary_url, code}`; a member with a code joins the primary |
| POST | `/api/fleet/join-code` | primary: a one-time join code (30 min) |
| POST | `/api/fleet/join` | primary, called by a member: `{code, box_id, name}` -> its token. No login (Caddy exempts it); the code is the proof |
| POST | `/api/fleet/report` | primary, called by members every 60 s with `Authorization: Bearer <token>`. No login; the token is the proof |
| POST | `/api/fleet/revoke` | primary: `{box_id}` |
| POST | `/api/fleet/send-now` | member: report now |
| POST | `/api/fleet/policy` | member: what the primary may do here `{allow: [instance.start, instance.stop, instance.restart, instance.set], instances: ["*"] or ids, allow_http}`; nothing by default |
| POST | `/api/fleet/action` | primary (admin): `{box_id, action, args: {instance, reason?, params?, restart?}}` -> a signed command, delivered with that box's next report; `instance.set` takes model and sizing settings only |
| POST | `/api/fleet/action/cancel` | primary: `{box_id, id}` a command the box has not taken yet |
| POST | `/api/fleet/drain` | primary: `{box_id, on}` the gateway sends a drained box nothing new |
| POST | `/api/fleet/rekey` | primary: `{box_id}` a new action key, sent with the box's next report |
| POST | `/api/fleet/event` | primary, called by members with their token: `{box_id, event: hold|release, instance, seconds}` around a planned restart of a shared instance. No login (Caddy exempts it); the token is the proof |

## Hermes Agent

| Method | Route | |
|---|---|---|
| GET | `/api/hermes` | what Hermes Agent needs of this llama.cpp instance (`--jinja`, 64k context per conversation, a template with tools, a reachable address), each as a check, and the `model:` and `mcp_servers:` blocks of `~/.hermes/config.yaml` |

## ONNX Runtime

| Method | Route | |
|---|---|---|
| GET | `/api/onnx/status` | the runtime Python's onnxruntime-genai version and which providers the build has, the NPUs the kernel sees, the provider list |

ONNX instances use the usual instance routes (`/api/params`, `/api/launch-plan`, `/api/start`, ...); their parameters are in `/api/param-meta`.

## Fit (hardware-fitted requants)

Host-wide; the instance a call measures or targets goes in the body as `instance`.
Planning, dry runs and builds never touch a server. `bench`, `imatrix` and `verify` stop the
instance they use and always start it again. One Fit job runs at a time.

| Method | Route | What |
|---|---|---|
| GET | `/api/fit/status` | readiness checklist, sources, card profiles, plans, jobs, candidates with verify results, disk, upstream watch |
| GET | `/api/fit/target` | `?instance=` the weight budget per card: today's weights + VRAM free now - margin |
| GET | `/api/fit/plan` | `?id=` one saved plan with every tensor's format |
| GET | `/api/fit/job` | `?id=` a job with the tail of its log |
| POST | `/api/fit/solve` | `{source, instance, goals:[faster,better,max], base, keep_free_mib, ctx, components:{output,mtp,embed}}` |
| POST | `/api/fit/check` | `{plan}` dry-run the plan through llama-quantize: exact size, overrides that landed |
| POST | `/api/fit/build` | `{plan, name, imatrix?, threads?}` quantize on the CPU at low priority |
| POST | `/api/fit/bench` | `{instance, source, types, depths, reps, keep_files}` measure formats on the instance's cards |
| POST | `/api/fit/imatrix` | `{instance, source, reference:q8_0\|current, tokens, calibration_files}` |
| POST | `/api/fit/verify` | `{instance, models, budget, curve_depths}` starts an optimizer run in models mode |
| POST | `/api/fit/convert` | `{kind:mmproj\|sd.cpp\|audio.cpp, input, type, rules, name}` |
| POST | `/api/fit/profile` | `{model, instance, name}` save a model profile that points at a candidate |
| POST | `/api/fit/delete` | `{path}` a candidate or test file (refused while any instance uses it) |
| POST | `/api/fit/cancel` | stop the running job |
| POST | `/api/fit/updates` | check upstream revisions now (also runs daily at 05:10 UTC) |
| POST | `/api/fit/source` | `{repo, revision?, name?}` phase A as a job: download and convert to BF16 |
| GET | `/api/fit/recipes` | recipes (built-in and imported), the models they can explain, the base-format menu |
| GET | `/api/fit/recipe` | `?id=` one recipe; `&export=1` for the portable JSON |
| POST | `/api/fit/explain` | `{model}` read a GGUF's tensors and explain the recipe that made it |
| POST | `/api/fit/recipe/check` | dry-run a recipe against a source: exact size and per-role formats |
| POST | `/api/fit/recipe/build` | build a model from a recipe (needs an importance matrix when the recipe does) |
| POST | `/api/fit/recipe/import` | `{recipe}` a recipe exported elsewhere |
| POST | `/api/fit/recipe/delete` | `{id}` (built-in recipes cannot be deleted) |
| POST | `/api/fit/imatrix/import` | `{path}` an importance-matrix file under the home folder |

## Refusal check (safety evaluation)

Per instance. Sends one chat request at a time to the instance's running server; nothing is
restarted or changed. Prompt sets download from Hugging Face on first use and are cached.

| Method | Route | What |
|---|---|---|
| GET | `/api/refusals/status` | `?inst=` the live run (if any), history with strict and keyword verdicts per set, defaults, marker lists |
| POST | `/api/refusals/start` | `{instance, n_harmful, n_harmless, thinking, max_tokens?}` start a check (one at a time) |
| POST | `/api/refusals/stop` | stop the running check |

## Models, downloads & templates

| Method | Route |
|---|---|
| GET | `/api/disk` |
| GET | `/api/downloads` |
| GET | `/api/hf-token` |
| GET | `/api/models` |
| GET | `/api/templates` |
| POST | `/api/download` |
| POST | `/api/hf-token` |
| POST | `/api/model/delete` |
| POST | `/api/template` |
| POST | `/api/template/extract` |

## Builds & engines

| Method | Route |
|---|---|
| GET | `/api/builds` |
| GET | `/api/builds/releases` |
| GET | `/api/engines` |
| GET | `/api/engines/releases` |
| POST | `/api/builds/activate` |
| POST | `/api/builds/delete` |
| POST | `/api/builds/install` |
| POST | `/api/engines/activate` |
| POST | `/api/engines/check` |
| POST | `/api/engines/delete` |
| POST | `/api/engines/install` |
| POST | `/api/engines/policy` |

## stable-diffusion.cpp

| Method | Route |
|---|---|
| GET | `/api/sd/gallery` |
| GET | `/api/sd/image` |
| GET | `/api/sd/job` |
| GET | `/api/sd/presets` |
| POST | `/api/sd/cancel` |
| POST | `/api/sd/download` |
| POST | `/api/sd/generate` |
| POST | `/api/sd/preset/apply` |

## audio.cpp

| Method | Route |
|---|---|
| GET | `/api/ac/catalog` |
| GET | `/api/ac/file` |
| GET | `/api/ac/job` |
| GET | `/api/ac/models` |
| GET | `/api/ac/outputs` |
| GET | `/api/ac/voices` |
| POST | `/api/ac/install` |
| POST | `/api/ac/models` |
| POST | `/api/ac/run` |
| POST | `/api/ac/sizes` |
| POST | `/api/ac/uninstall` |
| POST | `/api/ac/unload` |
| POST | `/api/ac/upload` |

## Camelid

| Method | Route |
|---|---|
| GET | `/api/cm/catalog` |
| GET | `/api/cm/local` |
| POST | `/api/cm/chat` |
| POST | `/api/cm/pull` |

## Files (home folder)

| Method | Route |
|---|---|
| GET | `/api/files/download` |
| GET | `/api/files/list` |
| GET | `/api/files/zip` |
| POST | `/api/files/delete` |
| POST | `/api/files/mkdir` |
| POST | `/api/files/paste` |
| POST | `/api/files/rename` |
| POST | `/api/files/upload` |

## GPU & power

| Method | Route |
|---|---|
| GET | `/api/gpu` |
| GET | `/api/gpu-power` |
| POST | `/api/gpu-power/clear` |
| POST | `/api/gpu-power/set` |

## Power options

Host-wide. Changes go through the root helper (`power/install-power.sh`); without it every
GET still answers and every change is refused with the install command.

| Method | Route | What |
|---|---|---|
| GET | `/api/power` | equipment, every setting (live / boot default / boot profile), drift, profiles, audit, helper state |
| GET | `/api/power/stability` | boots and how they ended, per profile (`?force=1` re-reads the journal) |
| GET | `/api/power/budget` | PSU / UPS ratings against the measured GPU peaks, UPS status |
| POST | `/api/power/apply` | `{settings:[{knob,target,ident,value}], label}` or `{profile}`: live, audited |
| POST | `/api/power/persist` | `{profile, apply_now?}` make it the boot profile |
| POST | `/api/power/unpersist` | no boot profile |
| POST | `/api/power/profile/save` | `{name, settings, description?}` |
| POST | `/api/power/profile/capture` | `{name, rows?}` save the live values |
| POST | `/api/power/profile/delete` | `{id}` |
| POST | `/api/power/settings` | `{auto_reapply, psu_w, ups_w, ...}` |
| POST | `/api/power/annotate` | `{boot_id, note: manual-off\|power-cut\|crash\|""}` |
| POST | `/api/power/recheck` | look for the helper again |

## GPU Tuning

Host-wide. Settings are the power helper's GPU knobs (`gpu.od_sclk`, `gpu.od_mclk`,
`gpu.od_voltage`, `gpu.power_cap`, `gpu.perf_level`, `gpu.fan`): range-checked by the helper,
read back, logged in the Power options audit, **live only** (a reboot restores stock). One
benchmark runs at a time, and never beside an optimizer or depth-curve run.

| Method | Route | What |
|---|---|---|
| GET | `/api/gpu-tune` | every card (ids, VBIOS, link, DPM tables, sensors, OverDrive table), its tuning settings, boot defaults, benchmark state and history, advice, VBIOS backups |
| GET | `/api/gpu-tune/bench` | the running benchmark (live samples) and recent results |
| GET | `/api/gpu-tune/vbios/file` | `?name=` download a VBIOS backup |
| POST | `/api/gpu-tune/apply` | `{settings:[{knob,target,ident,value}], label?}` GPU knobs only |
| POST | `/api/gpu-tune/bench/start` | `{instance, depth?, n_predict?, reps?, label?, trial?}`: with `trial` (settings) it applies them first and puts the previous values back if the run fails, the server dies, the kernel logs a GPU reset or a thermal limit trips |
| POST | `/api/gpu-tune/bench/stop` | |
| POST | `/api/gpu-tune/bench/delete` | `{id}` a saved result |
| POST | `/api/gpu-tune/profile/save` | `{name, settings:[{knob,target,ident,value}], description?}` save a fan/voltage/clock profile (stored with the Power options profiles) |
| POST | `/api/gpu-tune/profile/apply` | `{id}` apply a saved profile now (live) |
| POST | `/api/gpu-tune/profile/boot` | `{id, apply_now?}` make it the boot profile (applied before the inference servers start) |
| POST | `/api/gpu-tune/profile/unboot` | stop applying a profile at boot |
| POST | `/api/gpu-tune/profile/delete` | `{id}` |
| POST | `/api/gpu-tune/vbios/backup` | `{target: pci}` read the card's VBIOS through the helper (read only) into `panel/gpu-bios/` |
| POST | `/api/gpu-tune/rom-check` | `{target, path}` compare a ROM file (Files-tab path) with the card and its backup; returns checks, a verdict and the vendor tool's command. Never flashes |

## MCP (Model Context Protocol)

The panel's tools for AI clients; `mcp_server.py` has the tool list and the stdio mode.

| Method | Route | What |
|---|---|---|
| POST | `/api/mcp` | JSON-RPC 2.0 (MCP Streamable HTTP, JSON responses): `initialize`, `tools/list`, `tools/call`, `ping`; notifications get 202. GET answers 405 (no SSE stream) |
| GET | `/api/mcp/tools` | the same tool list as plain JSON |
| POST | `/api/mcp/call` | `{name, arguments}` run one tool, plain JSON |

`LEXIPANEL_MCP_READONLY=1` in the panel's environment hides every tool that changes something.

## llama.cpp built-in web UI

| Method | Route | What |
|---|---|---|
| GET | `/api/webui` | the instance's web UI state, its port and the Caddy route |
| POST | `/api/webui/defaults` | `{config}` the web UI's default settings |
| POST | `/api/webui/flags` | `{WEBUI, UI_MCP_PROXY}`: `on`, `off` or `""` |
| POST | `/api/webui/caddy-snippet` | write the snippet `webui/apply-caddy-webui.sh` reads |

## Diagnostics, crashes, logs & backup

| Method | Route |
|---|---|
| GET | `/api/backup` |
| GET | `/api/crash-report` |
| GET | `/api/debug-bundle` |
| GET | `/api/diagnostics` |
| GET | `/api/logs` |
| GET | `/api/stats` |

## Examples

```bash
# what is running
curl -s http://127.0.0.1:8090/api/servers | jq

# change a setting on an instance (takes effect at its next start)
curl -s http://127.0.0.1:8090/api/params?inst=main-test \
     -H 'Content-Type: application/json' -d '{"CTX": 65536}'

# preview exactly what a start would run
curl -s 'http://127.0.0.1:8090/api/launch-plan?inst=main-test' | jq '.argv, .errors, .warnings'

# audio.cpp: speak a line on an audio instance, then poll the job
curl -s 'http://127.0.0.1:8090/api/ac/run?inst=voice' -H 'Content-Type: application/json' \
     -d '{"model":"kokoro_82m_q8_0","text":"Hello there.","voice_id":"af_heart"}'
curl -s 'http://127.0.0.1:8090/api/ac/job?inst=voice&id=<job id>'

# Files tab: upload (raw body, NOT form data), list, download
curl -s 'http://127.0.0.1:8090/api/files/upload?dir=uploads&name=notes.txt' \
     -H 'Content-Type: application/octet-stream' --data-binary @notes.txt
curl -s 'http://127.0.0.1:8090/api/files/list?path=models' | jq '.entries[].name'
curl -s -OJ 'http://127.0.0.1:8090/api/files/download?path=uploads/notes.txt'

# Fit: plan against the running model, then build the "faster" candidate
curl -s http://127.0.0.1:8090/api/fit/solve -H 'Content-Type: application/json' \
     -d '{"source":"/home/admin/models/src/MyModel/MyModel-BF16.gguf","instance":"main","goals":["faster"]}' \
     | jq '.plans[] | {id, file_mib, predicted_decode, quality_index}'
curl -s http://127.0.0.1:8090/api/fit/build -H 'Content-Type: application/json' -d '{"plan":"<plan id>"}'

# GPU Tuning: benchmark at stock, then try +100 MHz memory with automatic revert
curl -s http://127.0.0.1:8090/api/gpu-tune/bench/start -H 'Content-Type: application/json' \
     -d '{"instance":"main","reps":5,"label":"stock"}'
curl -s http://127.0.0.1:8090/api/gpu-tune/bench | jq '.active | {state, verdict, summary}'
curl -s http://127.0.0.1:8090/api/gpu-tune/bench/start -H 'Content-Type: application/json' \
     -d '{"instance":"main","label":"mem 1350","trial":[{"knob":"gpu.od_mclk","target":"0000:03:00.0","value":"1350"}]}'

# Bench: is UBATCH=1024 really faster for my workload? (restarts the server, puts it back)
curl -s http://127.0.0.1:8090/api/bench/start -H 'Content-Type: application/json' \
     -d '{"kind":"compare","candidate":{"UBATCH":"1024"},"allow_restart":true}'
curl -s http://127.0.0.1:8090/api/bench | jq '.active | {step, sequential}'
curl -s 'http://127.0.0.1:8090/api/bench/traffic' | jq '{noise, requests_per_side}'

# MCP: list the tools, call one
curl -s http://127.0.0.1:8090/api/mcp -H 'Content-Type: application/json' \
     -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | jq '.result.tools[].name'
curl -s http://127.0.0.1:8090/api/mcp/call -H 'Content-Type: application/json' \
     -d '{"name":"get_status","arguments":{"instance":"main"}}' | jq .result

# GPU power cap (needs the udev grant, see README)
curl -s http://127.0.0.1:8090/api/gpu-power/set -H 'Content-Type: application/json' \
     -d '{"pci":"0000:03:00.0","watts":325}'
```
