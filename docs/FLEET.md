# LexiPanel Fleet — design

Full LexiPanel on every box; one box is the **primary** and sees them all.

**Status: built**: roles, one-time join codes, per-box tokens, reports every 60 s, the boxes list
with online / stale / offline, revoke, and serving shared models through the primary's gateway
with replicas and failover (1.0.0). **Remote actions (v2, 2026-09-26)**: from the primary, start,
stop, safely restart and re-configure (model and sizing) a member's instances, opt-in per box;
drain a box out of the gateway; a member's planned restart holds the primary's gateway for that
instance. **Not built yet**: the Alerts and Fits views, the hourly history, placement (picking
the boxes a model fits) and *Striping*.

## Goals and non-goals

- Every box keeps its full panel: it fits, tunes and auto-fits itself, as today.
- The primary shows the whole fleet in one tab and links to each box's own panel.
- Proven fits travel: a model × card result measured on one box is evidence for the next box
  with the same card.
- **Nothing changes on a member unless that member allows it.** Remote actions (*v2*) are off by
  default, allowed per action and per instance on the member itself, signed, and audited on both
  boxes. GPU, power and firmware settings stay local-only.
- No new daemon, database or dependency: stdlib Python inside the panel, files for state.

## Roles

One setting per box, `FLEET_ROLE` in `fleet/config.json`:

| Role | Behaviour |
|---|---|
| `standalone` | Default. Today's panel; no fleet traffic. |
| `primary` | Accepts reports from members, shows the Fleet tab, issues join codes, revokes members. |
| `member` | Sends a report to the primary every 60 s. Everything else as standalone. |

Any box can be promoted to primary (all are full installs). Failover = promote another box,
then point members at it (`fleet/config.json: primary_url`); no data is lost, each box keeps its own.

## Joining and revoking

1. Primary, Fleet tab: **Add a box** creates a one-time join code (random 128-bit, valid 30 min,
   single use), stored hashed in `fleet/joins.json`.
2. Member, Fleet settings: role `member`, the primary's URL (`https://primary/`, through its
   Caddy), and the code. The member POSTs `/api/fleet/join {code, box_id, name, pubinfo}`.
3. Primary checks the code, creates the box record, returns a per-box token (random 256-bit).
   The member stores it 0600 in `fleet/token`; the primary stores only its SHA-256.
4. **Revoke** on the primary deletes the token hash: the member's next report gets 401 and it
   shows "revoked by the primary" on its own Fleet card.

`box_id` is a random UUID made once per box (`fleet/box_id`), not the hostname.

## Transport

- Member → primary only: `POST /api/fleet/report` with `Authorization: Bearer <token>`,
  every 60 s (jitter ±10 s), over HTTPS through the primary's Caddy. Members open no port.
  Commands for a member ride back in the reply to its report; `POST /api/fleet/event` (same
  token) tells the primary a planned restart begins or ends. The primary never connects to a
  member.
- The report, join and event routes skip the panel's basic auth, and only these; each checks a
  box token (or a join code) itself. Caddy exempts exactly these three paths. Everything else on
  the primary stays behind login.
- Payload limit 256 KiB; rate limit 1 report / 20 s per box; body must be JSON.
- Offline members queue nothing: the next report is simply current. A box is **stale** after
  3 missed reports and **offline** after 10 minutes.

## The report (v1)

Built from the member's own existing routes, so it is only what the panel already knows:

```json
{
  "v": 1, "box_id": "…", "name": "rig-2", "sent": "2026-10-01T12:00:00Z",
  "lexipanel": "1.0.1", "os": "Ubuntu 26.04", "kernel": "6.18", "python": "3.14",
  "hardware": {"cpu": "…", "ram_mib": 65536,
               "gpus": [{"pci": "0000:03:00.0", "name": "RX 7900 XTX", "vram_mib": 24560, "driver": "amdgpu",
                         "temp_junction_c": 88, "power_w": 305, "power_cap_w": 327}],
               "npus": [{"vendor": "AMD", "driver": "amdxdna"}]},
  "instances": [{"id": "main", "engine": "llama.cpp", "state": "running", "model": "Qwen3.8-27B-IQ4_XS.gguf",
                 "backend": "vulkan", "ctx": 131072, "decode_tps": 38.1, "health": "ok"}],
  "workload": {"main": {"requests_per_day": 176, "depth_p90": 45091, "decode_p50": 37.5, "idle_hours": 105}},
  "autofit": {"main": {"mode": "propose", "pending_proposals": 1, "last": "applied UBATCH=1024, confirmed 108%"}},
  "curves": [{"model_sha": "…", "gpu": "RX 7900 XTX", "backend": "vulkan", "kv": "q8_0",
              "points": [[8192, 42.1], [32768, 35.0], [131072, 21.4]]}],
  "alerts": [{"kind": "crash", "at": "…", "text": "…"}, {"kind": "regression", "text": "…"}]
}
```

No prompts, no file contents, no settings files, no API keys: a box never sends what its own
workload profile does not already keep (timings, depths, counts).

## The primary's Fleet tab

- **Boxes**: name, online / stale / offline, LexiPanel version, GPUs / NPUs, instances with model
  and decode t/s, alerts count, pending auto-fit proposals, a link to that box's own panel.
- **Alerts**: every box's alerts in one list, newest first.
- **Fits**: decode-vs-depth by (model file hash × card × backend × KV type) across boxes, so a
  new box with a known card starts from a measured fit, not a guess. (The seed of a portable
  fit registry.)
- **Versions**: which boxes run an older LexiPanel.

State on the primary: `fleet/boxes/<box_id>.json` (last report + first/last seen) and
`fleet/boxes/<box_id>.jsonl` (hourly summaries, 60 days, trimmed like `workload/`).

## Security rules

- Tokens: 256-bit random, only hashes on the primary, 0600 on the member, never in argv or logs.
- Join codes: single use, 30 minutes, hashed at rest.
- The primary treats every report as untrusted data: size-limited, schema-checked, strings
  length-capped and HTML-escaped in the UI; unknown fields dropped.
- The primary never calls a member: `tests/test_safety.py` fails the build if fleet code makes
  any request other than a member posting to its primary's join, report or event route.
- A member can leave at any time (role back to `standalone`); the primary shows it offline
  until revoked.

## Remote actions (v2, built 2026-09-26)

The primary can act on a member's instances; each member decides what it accepts.

| Action | What the member does |
|---|---|
| `instance.start` | its own start path: RAM budget, start guard, launch plan checks |
| `instance.stop` | its own stop path |
| `instance.restart` | a **safe restart** (`restarts.py`): drain, restart, verify (settings + canary), fall back to the known-good settings when verify fails |
| `instance.set` | snapshot the settings, save the new ones, then a safe restart with that snapshot as the known-good: settings that do not verify are rolled back by the member itself and reported as failed |

`instance.set` changes model choice and sizing only (`fleet.SET_KEYS`: model, alias, context,
slots, KV type, batch sizes, offload, sampling defaults, the vLLM equivalents). Never the port,
bind address, API keys, backend, free-form arguments, the draft model (a full-size one locks the
host) or any path the server writes. A model must already be in the member's models folder: a
remote action never downloads or copies files, and never runs a shell command.

**Delivery.** The operator queues a command on the primary (`POST /api/fleet/action`, admin).
It is delivered in the reply to the member's next report (within about a minute; while commands
are pending or running the member reports every `FAST_S` = 22 s). The primary sends it with every
reply until the member acknowledges it; the member runs each id once. Results (`running`, then
`ok` / `failed` / `refused`, with the safe restart's outcome, downtime and held requests) come
back with the next report. A command not picked up within 10 minutes expires; a queued one can
be cancelled.

**Signing.** Each box has its own action key (256-bit), created by the primary and sent to the
member once, in the reply to the first report that allows an action. It is never sent again:
the primary stores only a hash of the report token, and a token seen on the wire must not be
enough to forge a command. A command is `{id, box_id, action, args, by, issued, expires}` signed
with HMAC-SHA256 over its canonical JSON. The member refuses one that is altered, signed with
another key, meant for another box, expired (5 minutes of clock skew allowed) or already seen.
Both boxes show the key's fingerprint; **Re-key** on the primary replaces a lost or suspect key.

**The member's rules** (`fleet/actions.json`, its Fleet tab): the actions it allows (none by
default), for which instances (`*` or ids), and whether it accepts commands from a plain-http
primary (off by default: with https only the primary can have sent them). The member re-checks
every command against its rules when it arrives, whatever the primary believed; it refuses one
while a Bench run, an optimizer or auto-fit run, or a restart of that instance is in progress,
and says why.

**Failures.** A lost acknowledgement: the primary sends the command again, the member ignores
the copy. The primary unreachable: results wait in the member's `fleet/outbox.json` for the next
report that gets through. The member's panel restarted mid-action: the action is reported
failed and never re-run (a safe restart it started is finished by `restarts.recover_on_startup`
and shows in the Safe restarts history). Every command, accepted or refused, is in the member's
`fleet/action-log.jsonl` and in both boxes' hash-chained audit logs.

**Drain.** `POST /api/fleet/drain {box_id, on}` on the primary: its gateway sends a drained
box nothing new (requests in flight finish), until undrained.

**Restart holds.** When a member runs a safe restart of an instance it shares, it tells the
primary first (`/api/fleet/event`, hold for the restart's hold + drain time, at most 75 min) and
again when done (release; also after a panel restart, since a lost release would otherwise hold
until it expires). Meanwhile the primary's gateway sends that instance's requests to another
replica of the model, or holds them when there is none, instead of failing them. Events are
token-checked and limited to 60 per box per 10 minutes.

**Measured** (2026-09-26, one box: a second panel as the primary, the box's own panel as the
member, a shared RTX 2060 instance serving Qwen3-1.7B, commands delivered by report):

| Command | Result |
|---|---|
| `instance.start` | ok; running and shared 38 s after it was queued |
| `instance.restart` under load (3 clients through the primary's gateway) | 210 requests, **all 200**; server down 8.0 s; the slowest request 9.9 s (held at the primary, not failed) |
| `instance.set` CTX=8192 | saved, restarted, verified: ok, 8.0 s down |
| `instance.set` MODEL outside the models folder | refused by the member; nothing changed |
| `instance.set` PORT | refused by the primary before sending |
| `instance.set` MODEL that does not load | the server never came up; the member put the known-good settings back by itself and reported "failed, recovered" (73 s down) |
| drain / undrain | the model left / rejoined the primary's `/v1/models` at once |
| `instance.stop` | ok |

## Striping one large model across boxes (next)

A model too big for any one box, split over several: llama.cpp's RPC backend. Each helper box
runs `rpc-server` (exposing its GPU or CPU memory); the head box's llama-server adds them with
`--rpc host1:port,host2:port` and places layers on them like local devices.

- On a member: an **RPC helper** instance type (engine llama.cpp, runs `rpc-server -H <lan ip>
  -p <port>`), its memory reported to the primary like any other instance.
- On the head: pick helpers from the fleet list; the launch plan adds `--rpc`, and the memory
  estimator counts the helpers' free VRAM / RAM as extra devices (tensor split by free memory).
- Guard rails: helpers listen on the LAN only, one head per helper, the plan refuses a helper
  that is offline or already in use; the link speed between boxes is shown, since decode is bound
  by it (a 1 GbE link makes large layers slow; 10 GbE+ recommended).
- Measured like everything else: the depth curve and the optimizer run on the striped server, so
  whether striping pays on this network is a number, not a hope.

## Files to add

| File | What |
|---|---|
| `fleet.py` | roles, join, report build (member) and intake (primary), stale / offline, trimming |
| `panel.py` | routes: `GET /api/fleet`, `POST /api/fleet/{settings,join-code,join,report,revoke}` |
| `static/index.html` | Fleet tab (primary) and Fleet card (member: role, primary URL, last report) |
| `systemd/Caddyfile.new` | the unauthenticated `/api/fleet/report`, `/api/fleet/join` and `/api/fleet/event` routes (token-checked by the panel) |
| `tests/test_fleet.py` | join (good, reused, expired code), report accepted / oversized / bad token / revoked, stale and offline, no secrets in reports |
| `tests/test_fleet_actions.py` | v2: key sent once, commands signed / run once / refused when altered, forged, misaddressed, late or replayed, the member's rules re-checked, settings limited to model and sizing, roll-back reported, drain, restart holds |
| `tests/e2e/fleet_run.sh` | a primary and three stand-in members on local ports: join, report, revoke, promote |

## Done when

- `bash tests/run_all.sh` passes with the new tests, and the e2e run shows three members joining,
  reporting, one revoked, and a member promoted to primary.
- README gains a *Fleet* section; CHANGELOG a release entry.
