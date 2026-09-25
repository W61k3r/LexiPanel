# LexiPanel Fleet — design

Full LexiPanel on every box; one box is the **primary** and sees them all.

**Status (1.0.0): built**: roles, one-time join codes, per-box tokens, reports every 60 s, the boxes
list with online / stale / offline, revoke, and serving shared models through the primary's
gateway with replicas and failover. **Not built yet**: the Alerts and Fits views, the hourly
history, deploying models from the primary, remote actions (*v2*) and *Striping*.

## Goals and non-goals

- Every box keeps its full panel: it fits, tunes and auto-fits itself, as today.
- The primary shows the whole fleet in one tab and links to each box's own panel.
- Proven fits travel: a model × card result measured on one box is evidence for the next box
  with the same card.
- **v1 is view-only across boxes.** The primary cannot start, stop, apply or change anything on
  a member. Remote actions are a later step (see *v2*), opt-in per box and audited.
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
- The report route skips the panel's basic auth only for this path and only with a valid box
  token; Caddy config gains that one route. Everything else on the primary stays behind login.
- Payload limit 256 KiB; rate limit 1 report / 20 s per box; body must be JSON.
- Offline members queue nothing: the next report is simply current. A box is **stale** after
  3 missed reports and **offline** after 10 minutes.

## The report (v1)

Built from the member's own existing routes, so it is only what the panel already knows:

```json
{
  "v": 1, "box_id": "…", "name": "rig-2", "sent": "2026-10-01T12:00:00Z",
  "lexipanel": "1.0.0", "os": "Ubuntu 26.04", "kernel": "6.18", "python": "3.14",
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
- v1 has no route by which the primary can change a member. Add a `tests/test_safety.py`
  tripwire: no fleet code calls the member's POST routes.
- A member can leave at any time (role back to `standalone`); the primary shows it offline
  until revoked.

## v2 (later, opt-in per box)

Remote actions from the primary (start / stop an instance, apply an auto-fit proposal, run a
depth curve): each enabled per box on the member itself, signed by the primary, re-checked by
the member's own guard rails, recorded in the member's audit log. GPU / power changes stay
local-only.

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
| `systemd/Caddyfile.new` | the unauthenticated `/api/fleet/report` and `/api/fleet/join` routes (token-checked by the panel) |
| `tests/test_fleet.py` | join (good, reused, expired code), report accepted / oversized / bad token / revoked, stale and offline, no secrets in reports |
| `tests/e2e/fleet_run.sh` | a primary and three stand-in members on local ports: join, report, revoke, promote |

## Done when

- `bash tests/run_all.sh` passes with the new tests, and the e2e run shows three members joining,
  reporting, one revoked, and a member promoted to primary.
- README gains a *Fleet* section; CHANGELOG a release entry.
