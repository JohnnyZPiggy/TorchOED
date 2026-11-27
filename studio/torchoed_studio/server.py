import argparse
import csv
import io
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from torchoed import central_composite, full_factorial


def handle(path: str, body: dict) -> dict | str:
    if path == "/api/export":
        out = io.StringIO(newline="")
        labels = [key for key in ("block", "whole_plot") if body.get(key) is not None]
        writer = csv.writer(out)
        writer.writerow(body["columns"] + labels)
        writer.writerows(row + [body[key][i] for key in labels] for i, row in enumerate(body["runs"]))
        return out.getvalue()
    if path != "/api/generate":
        raise ValueError("unknown endpoint")
    factors, design, model = body["factors"], body["design"], body["model"]
    if model not in ("linear", "quadratic"):
        raise ValueError("model must be linear or quadratic")
    names = [f["name"] for f in factors]
    terms = ["Intercept"] + names
    if model == "quadratic":
        terms += [f"{n}^2" for n in names] + [f"{a}:{b}" for i, a in enumerate(names) for b in names[i+1:]]
    runs, block, whole = [], [], None
    if design["kind"] == "central_composite":
        center = tuple(design["center"])
        coded = central_composite(len(factors), alpha=design["alpha"], center=center)
        runs = [[(f["low"]+f["high"])/2 + x*(f["high"]-f["low"])/2 for x, f in zip(row, factors)] for row in coded.tolist()]
        cut = 2**len(factors) + center[0]
        block = [1]*cut + [2]*(len(runs)-cut)
    elif design["kind"] == "split_plot_full_factorial":
        wp, sp = design["whole_plot"], design["subplot"]
        if set(wp + sp) != set(names) or len(wp + sp) != len(names):
            raise ValueError("whole_plot and subplot must partition factors")
        ordered = [next(f for f in factors if f["name"] == n) for n in wp + sp]
        grid = full_factorial([len(f["levels"]) for f in ordered]).tolist()
        whole, wp_ids = [], {}
        for b in range(design["blocks"]):
            for row in grid:
                values = {f["name"]: f["levels"][j] for f, j in zip(ordered, row)}
                runs.append([values[n] for n in names])
                block.append(b+1)
                key = (b, *row[:len(wp)])
                whole.append(wp_ids.setdefault(key, len(wp_ids)+1))
    else:
        raise ValueError("unknown design kind")
    return dict(columns=names, runs=runs, block=block, whole_plot=whole, model=model, terms=terms)


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        try:
            result = handle(self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            data = (result if isinstance(result, str) else json.dumps(result)).encode()
            status, content = 200, "text/csv" if isinstance(result, str) else "application/json"
        except (ValueError, KeyError, TypeError, RuntimeError) as error:
            data, status, content = json.dumps({"error": str(error)}).encode(), 400, "application/json"
        self.send_response(status)
        self.send_header("Content-Type", content + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    handler = partial(Handler, directory=str(Path(__file__).parent / "static"))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        server.serve_forever()
