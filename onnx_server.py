#!/usr/bin/env python3
"""
LexiPanel's ONNX Runtime GenAI server (added 2026-09-25): an OpenAI-compatible endpoint for an
ONNX model folder (genai_config.json + model.onnx), on the CPU or through an execution provider:
AMD (VitisAI: Ryzen AI NPU), Intel (OpenVINO: NPU / GPU / CPU), Qualcomm (QNN: Hexagon NPU),
NVIDIA (cuda), DirectML (dml, Windows), WebGPU. Run by LexiPanel's launcher; stdlib + onnxruntime_genai.

  GET  /health               {"status": "ok"} once the model is loaded
  GET  /v1/models            the model
  GET  /props                provider, context length, the effective search options and, for
                             each, whether this onnxruntime-genai build accepted it (set, then
                             read back): the parameter suite verified at every start
  POST /v1/chat/completions  messages, max_tokens, temperature, top_p, top_k, seed, stop, stream,
                             plus any search option by its own name
One request at a time (a lock); timings are printed per request.
"""
import argparse, ctypes, json, os, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SEARCH_TYPES = dict(max_length=int, min_length=int, do_sample=bool, temperature=float, top_k=int, top_p=float,
                    repetition_penalty=float, num_beams=int, num_return_sequences=int, length_penalty=float,
                    early_stopping=bool, no_repeat_ngram_size=int, diversity_penalty=float,
                    past_present_share_buffer=bool, batch_size=int, random_seed=int, chunk_size=int)
EP_NAMES = dict(cuda="cuda", dml="dml", openvino="OpenVINO", qnn="QNN", vitisai="VitisAI", webgpu="WebGPU")


def _name(n):
    try:
        ctypes.CDLL(None).prctl(15, n.encode(), 0, 0, 0)       # PR_SET_NAME: pgrep -x onnx-server
    except Exception:
        pass


def _typed(k, v):
    t = SEARCH_TYPES[k]
    if t is bool:
        return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
    return t(v)


def load(a):
    import onnxruntime_genai as og
    for spec in a.ep_library or []:
        name, _, path = spec.partition("=")
        og.register_execution_provider_library(name, path)
    cfg = og.Config(a.model_dir)
    if a.provider != "cpu":
        ep = EP_NAMES[a.provider]
        cfg.clear_providers()
        cfg.append_provider(ep)
        for k, v in json.loads(a.provider_options or "{}").items():
            cfg.set_provider_option(ep, str(k), str(v))
    elif a.provider_options and json.loads(a.provider_options):
        raise SystemExit("provider options need a provider other than cpu")
    so = json.loads(a.session_options or "{}")
    if so:
        cfg.overlay(json.dumps({"model": {"decoder": {"session_options": so}}}))
    model = og.Model(cfg)
    return og, model, og.Tokenizer(model)


def verify(og, model, wanted):
    """Each option set on its own, then read back: what this build really takes."""
    out = {}
    for k, v in wanted.items():
        try:
            p = og.GeneratorParams(model)
            p.set_search_options(**{k: v})
            got = p.get_search_options().get(k)
            ok = got is not None and (abs(float(got) - float(v)) < 1e-4 * max(1, abs(float(v))))
            out[k] = dict(ok=ok, value=v, read_back=got)
        except Exception as e:
            out[k] = dict(ok=False, value=v, error=str(e)[:200])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8087)
    ap.add_argument("--provider", default="cpu", choices=["cpu"] + list(EP_NAMES))
    ap.add_argument("--provider-options", default="{}", help="JSON, e.g. {\"device_type\": \"NPU\"}")
    ap.add_argument("--session-options", default="{}", help="JSON, e.g. {\"intra_op_num_threads\": 6}")
    ap.add_argument("--ep-library", action="append", help="NAME=PATH of a plugin execution provider library")
    ap.add_argument("--search", default="{}", help="JSON of default search options")
    ap.add_argument("--alias", default="")
    ap.add_argument("--api-key-file", default="")
    a = ap.parse_args()
    _name("onnx-server")
    cfg_file = json.load(open(os.path.join(a.model_dir, "genai_config.json")))
    ctx = int((cfg_file.get("model") or {}).get("context_length") or 0)
    defaults = dict(cfg_file.get("search") or {})
    for k, v in json.loads(a.search or "{}").items():
        if k not in SEARCH_TYPES:
            raise SystemExit(f"not a search option: {k}")
        defaults[k] = _typed(k, v)
    defaults = {k: _typed(k, v) for k, v in defaults.items() if k in SEARCH_TYPES}
    t0 = time.time()
    og, model, tok = load(a)
    checked = verify(og, model, defaults)
    bad = [k for k, r in checked.items() if not r["ok"]]
    print(f"onnx-server: {a.model_dir} on {a.provider} loaded in {time.time() - t0:.1f} s; context {ctx}; "
          f"{len(checked) - len(bad)}/{len(checked)} search options verified"
          + (f"; NOT accepted: {', '.join(bad)}" if bad else ""), flush=True)
    key = open(a.api_key_file).read().strip() if a.api_key_file else ""
    name = a.alias or os.path.basename(os.path.normpath(a.model_dir))
    lock = threading.Lock()
    state = dict(busy=False, served=0)

    def generate(body, emit):
        msgs = body.get("messages") or []
        tools = body.get("tools")
        prompt = tok.apply_chat_template(json.dumps(msgs), tools=json.dumps(tools) if tools else None,
                                         add_generation_prompt=True)
        ids = tok.encode(prompt)
        opts = {k: v for k, v in defaults.items() if checked.get(k, {}).get("ok")}
        for k in SEARCH_TYPES:
            if k in body:
                opts[k] = _typed(k, body[k])
        for src, dst in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"), ("seed", "random_seed")):
            if body.get(src) is not None:
                opts[dst] = _typed(dst, body[src])
        want = int(body.get("max_tokens") or body.get("max_completion_tokens") or 512)
        limit = ctx or int(opts.get("max_length") or 4096)
        if len(ids) >= limit:
            raise ValueError(f"the prompt is {len(ids)} tokens; this model holds {limit}")
        opts["max_length"] = min(limit, len(ids) + want)
        p = og.GeneratorParams(model)
        p.set_search_options(**opts)
        g = og.Generator(model, p)
        stops = body.get("stop") or []
        stops = [stops] if isinstance(stops, str) else stops
        t1 = time.time()
        g.append_tokens(ids)
        t2 = time.time()
        stream, text, n, reason = tok.create_stream(), "", 0, "length"
        while not g.is_done():
            g.generate_next_token()
            piece = stream.decode(g.get_next_tokens()[0])
            n += 1
            text += piece
            hit = next((s for s in stops if s and s in text), None)
            if hit:
                text = text[:text.index(hit)]
                reason = "stop"
                break
            emit(piece)
        else:
            reason = "stop" if n < want else "length"
        t3 = time.time()
        tim = dict(prompt_n=len(ids), prompt_ms=round((t2 - t1) * 1000, 1),
                   prompt_per_second=round(len(ids) / max(t2 - t1, 1e-6), 2), predicted_n=n,
                   predicted_ms=round((t3 - t2) * 1000, 1), predicted_per_second=round(n / max(t3 - t2, 1e-6), 2))
        print(f"onnx-server: prompt {tim['prompt_n']} tokens {tim['prompt_per_second']} t/s, "
              f"{n} generated at {tim['predicted_per_second']} t/s", flush=True)
        return text, reason, tim

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *x):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _authed(self):
            if not key or self.headers.get("Authorization", "") == f"Bearer {key}":
                return True
            self._send(401, dict(error=dict(message="invalid API key")))
            return False

        def do_GET(self):
            if self.path.startswith("/health"):
                return self._send(200, dict(status="ok"))
            if not self._authed():
                return
            if self.path.startswith("/v1/models"):
                return self._send(200, dict(object="list", data=[dict(id=name, object="model", owned_by="lexipanel")]))
            if self.path.startswith("/props"):
                return self._send(200, dict(engine="onnx", model_dir=a.model_dir, provider=a.provider,
                                            context_length=ctx, busy=state["busy"], served=state["served"],
                                            search=defaults, verified=checked))
            self._send(404, dict(error=dict(message="not found")))

        def do_POST(self):
            if not self._authed():
                return
            if not self.path.startswith("/v1/chat/completions"):
                return self._send(404, dict(error=dict(message="not found")))
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except ValueError:
                return self._send(400, dict(error=dict(message="body is not JSON")))
            rid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
            chunk = lambda delta, fin=None: dict(id=rid, object="chat.completion.chunk", created=created, model=name,
                                                 choices=[dict(index=0, delta=delta, finish_reason=fin)])
            with lock:
                state["busy"] = True
                try:
                    if body.get("stream"):
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        sse = lambda o: (self.wfile.write(f"data: {json.dumps(o)}\n\n".encode()), self.wfile.flush())
                        sse(chunk(dict(role="assistant")))
                        text, reason, tim = generate(body, lambda s: s and sse(chunk(dict(content=s))))
                        sse(dict(chunk({}, reason), timings=tim))
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.close_connection = True
                    else:
                        text, reason, tim = generate(body, lambda s: None)
                        self._send(200, dict(id=rid, object="chat.completion", created=created, model=name,
                                             choices=[dict(index=0, finish_reason=reason,
                                                           message=dict(role="assistant", content=text))],
                                             usage=dict(prompt_tokens=tim["prompt_n"], completion_tokens=tim["predicted_n"],
                                                        total_tokens=tim["prompt_n"] + tim["predicted_n"]),
                                             timings=tim))
                    state["served"] += 1
                except (ValueError, KeyError, TypeError) as e:
                    self._send(400, dict(error=dict(message=str(e))))
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    state["busy"] = False

    print(f"onnx-server: listening on {a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    sys.exit(main())
