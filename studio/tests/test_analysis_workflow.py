import json
from pathlib import Path
from threading import Thread
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer
import pytest
import torch
from torchoed import fit_least_squares, fit_reml, predict_least_squares, prediction_trace, d_max, maximize_desirability
from torchoed_studio.server import Handler, handle, json_values

ROOT = Path(__file__).resolve().parents[2]
E1 = [dict(name="Time", low=80, high=90), dict(name="Temp", low=170, high=180)]
E2 = [dict(name="Variety", levels=["Victory", "Golden Rain", "Marvellous"]), dict(name="nitro", levels=[0, .2, .4, .6])]

def imported(oats=False, http=False):
    request = dict(csv=(ROOT/("reports/w17-reml-check/Oats.csv" if oats else "reports/w17-lm-published/ChemReact.csv")).read_text(), factors=E2 if oats else E1, response="yield" if oats else "Yield", block="Block", whole_plot=["Block", "Variety"] if oats else None)
    if http:
        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = Thread(target=server.serve_forever); thread.start()
            try:
                with urlopen(Request(f"http://127.0.0.1:{server.server_port}/api/import", data=json.dumps(request).encode())) as response:
                    out = json.load(response)
            finally:
                server.shutdown(); thread.join()
    else:
        out = handle("/api/import", request)
    return dict(out, factors=request["factors"], model="interaction" if oats else "quadratic")

def chem_rows(x, blocks=None):
    x = torch.as_tensor(x, dtype=torch.float64)
    b = x.new_zeros(len(x)) if blocks is None else torch.tensor(blocks)-1
    return torch.stack((torch.ones_like(x[:, 0]), b, x[:, 0], x[:, 1], x[:, 0]**2, x[:, 1]**2, x[:, 0]*x[:, 1]), 1)

def chem_fit(body):
    x = [[(a-85)/5, (b-175)/5] for a, b in body["runs"]]
    return fit_least_squares(chem_rows(x, body["block"]), torch.tensor(body["response"], dtype=torch.float64), terms=(1,)*7)

def close(actual, expected, tol, absolute=False):
    a, b = [torch.as_tensor(v, dtype=torch.float64).reshape(-1) for v in (actual, expected)]
    assert torch.all((a-b).abs() <= tol if absolute else (a-b).abs()/b.abs().clamp_min(1.) <= tol)

def test_chemreact_fit():
    body = imported(http=True)
    out = handle("/api/fit", body)
    assert out == dict(json_values(chem_fit(body)), kind="least_squares", terms=out["terms"])
    ref = json.loads((ROOT/"reports/w17-lm-published/r.json").read_text())["ChemReact"]
    for key in ("coef", "se", "t", "residuals", "leverage", "p_t"):
        close(out[key], ref[key], 1e-8, key == "p_t")
    for group, keys in (("anova", ("ss", "f", "p")), ("lack_of_fit", ("f", "p"))):
        for key in keys:
            close(out[group][key], ref[group][key], 1e-8, key == "p")

def test_oats_fit():
    body = imported(True)
    matrix = [[1]+[float(v == z) for z in ("Marvellous", "Victory")]+[float(n == z) for z in (.2, .4, .6)]+[float(n == z and v == w) for z in (.2, .4, .6) for w in ("Marvellous", "Victory")] for v, n in body["runs"]]
    fit = fit_reml(torch.tensor(matrix, dtype=torch.float64), torch.tensor(body["response"], dtype=torch.float64), tuple(torch.tensor(body[k]) for k in ("block", "whole_plot")))
    out = handle("/api/fit", body)
    assert out == dict(json_values(fit), kind="reml", terms=out["terms"])
    ref = json.loads((ROOT/"reports/w17-reml-check/results.json").read_text())["cases"]["Oats"]["exact"]
    for a, b in (("coef", "fixed"), ("se", "SE"), ("df", "df"), ("variance", "variance"), ("loglik", "logLik")):
        close(out[a], ref[b], 1e-10)
    with pytest.raises(ValueError, match="profile needs a least-squares fit"):
        handle("/api/profile", body)

def test_chemreact_profile():
    body = imported(); fit = chem_fit(body)
    ref = json.loads((ROOT/"reports/w18-r-refs/r.json").read_text())["ChemReact"]
    for i, point in enumerate(ref["points"]):
        x = [point["x1"], point["x2"]]
        out = handle("/api/profile", dict(body, point=[85+5*x[0], 175+5*x[1]], prediction_block=1))["prediction"]
        natural_x = [[((85+5*x[0])-85)/5, ((175+5*x[1])-175)/5]]
        assert out == json_values(predict_least_squares(fit, chem_rows(natural_x)))
        close(out["fit"], ref["confidence"]["fit"][i][0], 1e-8)
        close(out["se"], ref["confidence"]["se.fit"][i], 1e-8)
        for key, name in (("ci", "confidence"), ("pi", "prediction")):
            close(out[key], ref[name]["fit"][i][1:], 1e-8)
    out = handle("/api/profile", dict(body, trace=dict(factor="Time", grid=[80, 85, 90]), goal=dict(kind="max", low=80, high=84)))
    assert out["trace"] == json_values(prediction_trace(fit, chem_rows, torch.zeros(2, dtype=torch.float64), 0, torch.tensor([-1., 0., 1.], dtype=torch.float64)))
    assert out["prediction"] == json_values(predict_least_squares(fit, chem_rows([[0., 0.]])))
    assert handle("/api/profile", dict(body, prediction_block=2))["prediction"] == json_values(predict_least_squares(fit, chem_rows([[0., 0.]], [2])))
    bounds = torch.ones(2, dtype=torch.float64)
    assert out["goal"] == json_values(maximize_desirability(lambda x: d_max(predict_least_squares(fit, chem_rows(x))["fit"], 80, 84), -bounds, bounds))
