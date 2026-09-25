#!/usr/bin/env python3
"""A stand-in llama-server process for the auto-fit end-to-end demo. Takes llama-server's
flags (so /proc/<pid>/cmdline looks like one to the panel), answers /health, /props and
/slots. A slot reads as busy while the file in $FAKE_BUSY_FILE exists."""
import http.server, json, os, socketserver, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import speed_model as SM  # noqa: E402

argv = sys.argv[1:]
port = int(argv[argv.index("--port") + 1])
ctx = int(argv[argv.index("-c") + 1]) if "-c" in argv else 131072
busy_file = os.environ.get("FAKE_BUSY_FILE", "")
log_file = argv[argv.index("--log-file") + 1]
params = SM.params_of(argv)
task = [0]


def log(lines):
    with open(log_file, "a") as f:
        f.write("".join(l + "\n" for l in lines))


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/tokenize":
            return self._send(dict(tokens=list(range(len(body.get("content", "")) // 4))))
        depth = len(body.get("prompt") or []) or 300
        n = int(body.get("n_predict") or body.get("max_tokens") or 16)
        dec, pre = SM.decode(depth, params), SM.prefill(depth, params)
        time.sleep(0.2)
        task[0] += 1
        log(SM.log_lines(depth + n, depth, n, dec, pre, task[0]))
        if self.path == "/completion":
            return self._send(dict(content="x" * n, timings=dict(predicted_n=n, predicted_per_second=dec,
                                   prompt_n=depth, prompt_per_second=pre, draft_n=100, draft_n_accepted=71)))
        return self._send(dict(choices=[dict(message=dict(role="assistant", content="OK"))],
                               timings=dict(predicted_n=n, predicted_per_second=dec, prompt_n=depth,
                                            prompt_per_second=pre)))

    def do_GET(self):
        if self.path.startswith("/slots"):
            obj = [dict(id=0, n_ctx=ctx, is_processing=bool(busy_file and os.path.exists(busy_file)))]
        elif self.path.startswith("/props"):
            obj = dict(default_generation_settings=dict(n_ctx=ctx))
        else:
            obj = dict(status="ok")
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


class T(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


T(("127.0.0.1", port), H).serve_forever()
