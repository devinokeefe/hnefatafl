"""Play against the engine in a browser.

    python -m tafl.play --model runs/r1/checkpoints/step_0100000.pt [--port 8080]

Then open http://localhost:8080. The page talks to two JSON endpoints:
POST /api/state {"moves": [...]} returns the position (legal moves, result) and
POST /api/think {"moves": [...], "movetime_ms": 1000} returns the engine's move.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch

from .inference import Batch, Evaluator
from .model import load_model
from .native import Pool, TaflError, game_info

WEB = Path(__file__).with_name("web")
MAX_MOVETIME_MS = 60_000


class Engine:
    def __init__(self, model_path: str, device: torch.device, leaves: int = 8) -> None:
        model, data = load_model(model_path, device)
        self.name = f"{Path(model_path).stem} (step {data.get('steps', '?')})"
        net = model.fused() if device.type == "cuda" else model.eval()
        self.evaluator = Evaluator(net, device)
        self.pool = Pool(
            games=1, autoplay=False, leaves_per_search=leaves, search={"algorithm": "puct"}, reuse_tree=True
        )
        self.batch = Batch(self.pool.max_batch, device)
        self.lock = threading.Lock()

    def think(self, moves: list[str], movetime_ms: int) -> dict[str, Any]:
        with self.lock:
            self.pool.set_game(0, "startpos", " ".join(moves))
            self.pool.start_search(0, 50_000_000)
            start = time.perf_counter()
            while self.pool.is_thinking(0):
                if (time.perf_counter() - start) * 1000 >= movetime_ms:
                    self.pool.stop_search(0)
                rows = self.pool.collect(self.batch.planes_np)
                self.evaluator.launch(self.batch, rows)
                self.evaluator.wait(self.batch)
                if rows:
                    self.pool.apply(self.batch.policy_np, self.batch.value_np, rows)
            result = self.pool.state(0)["result"]
            stats = sorted(result["stats"], key=lambda s: -s["visits"])[:8]
            return {
                "best": result["best"],
                "value": result["value"],
                "proof": result["proof"],
                "pv": result["pv"],
                "simulations": result["simulations"],
                "time_ms": round((time.perf_counter() - start) * 1000),
                "candidates": stats,
            }


def make_handler(engine: Engine) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # quiet
            pass

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, value: Any) -> None:
            self._send(code, json.dumps(value).encode(), "application/json")

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/engine":
                self._json(200, {"name": engine.name})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length) or b"{}")
                moves = [str(m) for m in request.get("moves", [])]
                if self.path == "/api/state":
                    self._json(200, game_info("startpos", " ".join(moves)))
                elif self.path == "/api/think":
                    info = game_info("startpos", " ".join(moves))
                    if info["outcome"] is not None:
                        raise TaflError("the game is over")
                    movetime = min(max(int(request.get("movetime_ms", 1000)), 10), MAX_MOVETIME_MS)
                    self._json(200, engine.think(moves, movetime))
                else:
                    self._json(404, {"error": "not found"})
            except (TaflError, ValueError, TypeError) as e:
                self._json(400, {"error": str(e)})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--leaves", type=int, default=8, help="leaves evaluated together per search step")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = Engine(args.model, device, args.leaves)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine))
    print(f"playing with {engine.name} on http://localhost:{args.port}", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()
