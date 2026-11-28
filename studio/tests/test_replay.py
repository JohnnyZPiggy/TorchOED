import json
import math
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.request import urlopen
from runpy import run_path

recorder = run_path(str(Path(__file__).resolve().parents[2] / 'tools/studio_replay.py'))
OUTPUT, record = recorder['OUTPUT'], recorder['record']
from torchoed_studio.server import Handler

REPLAY_TOL = 1e-10


def close(a, b, path=''):
    if type(a) is float or type(b) is float:
        ok = isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=REPLAY_TOL, abs_tol=REPLAY_TOL)
        return [] if ok else [(path, a, b)]
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        return [bad for k in a for bad in close(a[k], b[k], f'{path}/{k}')]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [bad for i, (x, y) in enumerate(zip(a, b)) for bad in close(x, y, f'{path}/{i}')]
    return [] if type(a) is type(b) and a == b else [(path, a, b)]


def test_recordings_and_regeneration():
    original = OUTPUT.read_bytes()
    entries = json.loads(original)
    assert set(entries) == {f'{example} /api/{step}' for example in ('ChemReact', 'Oats')
                           for step in ('generate', 'evaluate', 'compare', 'export', 'import', 'fit', 'profile')
                           if (example, step) != ('Oats', 'profile')}
    fresh = json.loads(record())
    assert {key: entry['request'] for key, entry in fresh.items()} == {key: entry['request'] for key, entry in entries.items()}
    bad = close(fresh, entries)
    assert not bad, bad


def test_static_assets_http():
    with ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(OUTPUT.parent))) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            for name in ('manifest.webmanifest', 'sw.js', 'icon.svg'):
                with urlopen(f'http://127.0.0.1:{server.server_port}/{name}') as response:
                    assert response.status == 200
                    assert response.read() == (OUTPUT.parent / name).read_bytes()
        finally:
            server.shutdown()
            thread.join()

