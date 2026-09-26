# LexiPanel roadmap (after 1.0.0)

## Built in 1.0.0

- **Users, roles and API keys**: single- or multi-user mode; viewer / operator / admin; API keys
  with expiry; MCP calls with the caller's role; hash-chained audit log (`auth.py`).
- **Gateway and quotas**: one OpenAI-compatible `/v1` endpoint for every instance, per-user
  quotas, usage per day, user and model (`gateway.py`).
- **Fleet**: one primary, many boxes; join codes, per-box tokens, reports, revoke; shared models
  served by the primary's gateway with replicas and failover (`fleet.py`, `docs/FLEET.md`).

## Next

1. **Quotas and usage on screen** (today API only: `POST /api/gateway/quota`, `GET /api/gateway`).
2. **Deploy a model to the fleet** from the primary. *Built 2026-09-26*: Fleet v2 remote actions
   (start, stop, safe restart, model and sizing changes with roll-back; signed, opt-in per box,
   re-checked by the member's own guard rails, audited on both sides), drain, and restart holds
   at the primary's gateway. *Next*: placement - pick the boxes that fit (free VRAM / RAM, card,
   measured speed of that model x card) and start N replicas. GPU and power changes stay local-only.
3. **Striping** a model too big for one box across several with llama.cpp RPC (`docs/FLEET.md`),
   measured before it is trusted.
4. **Fleet views**: alerts from every box, the model x card fits table, hourly history.
5. **SSO**: OIDC through Caddy mapped to the same roles; MFA through it.
6. **SOC 2 readiness** (below).

## SOC 2 readiness

SOC 2 is an audit of an **organization's** controls by a CPA firm against the AICPA Trust Services
Criteria, not a property of software. LexiPanel can make the controls an operator needs easy to
run and to evidence. Type I tests their design at one date; Type II that they worked over an
observation period (usually 3-12 months), so evidence must be kept as it happens.

| Criterion | LexiPanel 1.0.0 | Still to build |
|---|---|---|
| CC6.1-6.3 Logical access | Multi-user mode: users, roles, API keys with expiry, removal | SSO, scheduled access reviews |
| CC6.1 Authentication | Panel logins (scrypt); Caddy login in front of the terminals | Password policy, MFA via SSO |
| CC6.6-6.7 Transmission | Caddy TLS; raw engine ports LAN-only; LAN listeners need a key (ONNX, Camelid) | Public-CA TLS guidance |
| CC6.1 Data at rest | Secrets in 0600 files, never in argv; token and key hashes only | Disk-encryption guidance, secrets inventory |
| CC7.2 Monitoring | Hash-chained audit log of every change; power options audit; crash forensics; fleet status | Alerts to email / webhook |
| CC7.3-7.4 Incidents | Crash triage, debug bundle | Incident records linked to logs |
| CC8.1 Change management | CHANGELOG, `tests/run_all.sh`, CI on every push, safety tripwires, auto-fit verify-and-rollback | Signed release tags, required review before merge |
| CC7.1 Vulnerabilities | Stdlib only; upstream builds SHA-256-checked where published | Provenance record, CVE check for engines and Caddy |
| A1.2 Availability | `make-backup.sh`, fallback tiers, start guard, gateway failover | Scheduled backups, restore test, recovery target |
| C1.1 / P Confidentiality, privacy | No prompts stored; counts and timings only; 60-day retention | Retention settings per data type, data inventory |
| CC9.2 Vendors | None at runtime; Hugging Face and GitHub for downloads | Subprocessor list |

**The operator's part** (outside the software): written policies (security, access, change,
incident, backup, vendor, retention), a risk assessment, training, access reviews, a named owner
per control, and the auditor.

**Evidence bundle** (to build): an admin-only, scheduled, kept archive with users and roles, key
metadata (never secrets), the audit log and its chain check, change and CI records per release,
backup and restore tests, retention settings, provenance, and the fleet's box list.
