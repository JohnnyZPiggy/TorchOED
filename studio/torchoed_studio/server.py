import argparse
import csv
import io
import json
from functools import partial
from itertools import groupby
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from torchoed import (central_composite, full_factorial, effects_coding,
                      polynomial_features, design_diagnostics, compare_designs,
                      fit_least_squares, fit_reml, predict_least_squares,
                      prediction_trace, d_max, maximize_desirability)


def features(body):
    columns = []
    for j, factor in enumerate(body["factors"]):
        values = [row[j] for row in body["runs"]]
        if "low" in factor:
            x = torch.tensor(values, dtype=torch.float64)
            columns.append(((x - (factor["low"] + factor["high"]) / 2)
                            / ((factor["high"] - factor["low"]) / 2))[:, None])
        else:
            codes = torch.tensor([factor["levels"].index(v) for v in values])
            columns.append(effects_coding(codes, len(factor["levels"])))
    coded = torch.cat(columns, dim=1)
    if body["model"] == "quadratic":
        return polynomial_features(coded, 2)
    return torch.cat((torch.ones((len(coded), 1), dtype=torch.float64), coded), dim=1)


def fit_matrix(body):
    groups, names = [], ["Intercept"]
    n = len(body["runs"])
    columns = [torch.ones((n, 1), dtype=torch.float64)]
    if body.get("whole_plot") is None and body.get("block") is not None:
        ids = torch.tensor(body["block"])
        k = int(ids.max())
        columns += [(ids[:, None] == torch.arange(2, k+1)).double()]
        names += [f"Block{b}" for b in range(2, k+1)]
    for j, factor in enumerate(body["factors"]):
        values = [row[j] for row in body["runs"]]
        if "low" in factor:
            x = torch.stack([torch.as_tensor(v, dtype=torch.float64) for v in values])
            group = ((x-(factor["low"]+factor["high"])/2)/((factor["high"]-factor["low"])/2))[:, None]
            labels = [factor["name"]]
        else:
            levels = sorted(factor["levels"])[1:]
            group = torch.tensor([[v == level for level in levels] for v in values], dtype=torch.float64)
            labels = [f"{factor['name']}[{v}]" for v in levels]
        groups.append((group, labels))
        columns.append(group)
        names += labels
    if body["model"] == "quadratic":
        columns += [g.square() for g, _ in groups]
        names += [f"{labels[0]}^2" for _, labels in groups]
    if body["model"] in ("quadratic", "interaction"):
        for i, (a, an) in enumerate(groups):
            for b, bn in groups[i+1:]:
                columns += [(a[:, :, None]*b[:, None, :]).transpose(1, 2).flatten(1)]
                names += [f"{x}:{y}" for y in bn for x in an]
    body["terms"] = names
    return torch.cat(columns, dim=1)


def json_values(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {key: json_values(v) for key, v in value.items()}
    if isinstance(value, list):
        return [json_values(v) for v in value]
    return value


def handle(path: str, body: dict) -> dict | str:
    if path == "/api/import":
        records = list(csv.DictReader(io.StringIO(body["csv"])))
        def ids(keys):
            labels = [tuple(r[k] for k in keys) for r in records]
            order = list(dict.fromkeys(labels))
            return [order.index(v)+1 for v in labels]
        factors = body["factors"]
        runs = [[float(r[f["name"]]) if "low" in f or all(isinstance(v, (int, float)) for v in f["levels"]) else r[f["name"]] for f in factors] for r in records]
        return dict(columns=[f["name"] for f in factors], runs=runs,
                    response=[float(r[body["response"]]) for r in records],
                    block=ids([body["block"]]) if body.get("block") else None,
                    whole_plot=ids(body["whole_plot"]) if body.get("whole_plot") else None)
    if path in ("/api/fit", "/api/profile"):
        F, y = fit_matrix(body), torch.tensor(body["response"], dtype=torch.float64)
        reml = body.get("whole_plot") is not None
        fit = fit_reml(F, y, tuple(torch.tensor(body[k]) for k in ("block", "whole_plot"))) if reml else fit_least_squares(F, y, terms=(1,)*F.shape[1])
        if path == "/api/fit":
            return dict(json_values(fit), kind="reml" if reml else "least_squares", terms=body["terms"])
        if reml:
            raise ValueError("profile needs a least-squares fit")
        factors = body["factors"]
        point = body.get("point", [(f["low"]+f["high"])/2 if "low" in f else sorted(f["levels"])[0] for f in factors])
        k = max(body["block"]) if body.get("block") else 1
        def rows(coded):
            factors_coded = [dict(f, low=-1, high=1) if "low" in f else f for f in factors]
            block = [body.get("prediction_block", 1) if k > 1 else 1]*len(coded)+[k] if body.get("block") else None
            # One extra row retains every training block dummy, including for B1 predictions.
            return fit_matrix(dict(body, factors=factors_coded, runs=list(coded)+[f0], block=block))[:len(coded)]
        f0 = [((v-(f["low"]+f["high"])/2)/((f["high"]-f["low"])/2)) if "low" in f else v for v, f in zip(point, factors)]
        level = body.get("level", .95)
        out = dict(prediction=predict_least_squares(fit, rows([f0]), level=level))
        if body.get("trace"):
            trace = body["trace"]
            j = next(i for i, f in enumerate(factors) if f["name"] == trace["factor"])
            f = factors[j]
            grid = (torch.tensor(trace["grid"], dtype=torch.float64)-(f["low"]+f["high"])/2)/((f["high"]-f["low"])/2)
            out["trace"] = prediction_trace(fit, rows, torch.tensor(f0, dtype=torch.float64), j, grid, level=level)
        if body.get("goal"):
            goal = body["goal"]
            if goal["kind"] != "max":
                raise ValueError("goal kind must be max")
            bounds = torch.ones(len(factors), dtype=torch.float64)
            out["goal"] = maximize_desirability(lambda x: d_max(predict_least_squares(fit, rows(x))["fit"], goal["low"], goal["high"]), -bounds, bounds)
        return json_values(out)
    if path in ("/api/evaluate", "/api/compare"):
        F = features(body)
        ids = torch.arange(len(F))
        options = dict(effect=body.get("effect", 1.0), alpha=body.get("alpha", 0.05))
        if path == "/api/compare":
            return json_values(compare_designs(F, [ids, len(F)], **options,
                               generator=torch.Generator().manual_seed(body.get("seed", 0))))
        if body.get("whole_plot") is not None:
            options.update(whole_plots=[len(list(g)) for _, g in groupby(body["whole_plot"])], eta=1.0)
        elif len(set(body["block"])) > 1:
            options["blocks"] = [len(list(g)) for _, g in groupby(body["block"])]
            F = F[:, 1:]
        return json_values(design_diagnostics(F, ids, **options))
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
