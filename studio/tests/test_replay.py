import copy
import json
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.request import urlopen
from runpy import run_path

recorder = run_path(str(Path(__file__).resolve().parents[2] / 'tools/studio_replay.py'))
OUTPUT, record = recorder['OUTPUT'], recorder['record']
from torchoed_studio.server import Handler, handle


def test_recordings_and_regeneration():
    original = OUTPUT.read_bytes()
    entries = json.loads(original)
    assert set(entries) == {f'{example} /api/{step}' for example in ('ChemReact', 'Oats')
                           for step in ('generate', 'evaluate', 'compare', 'export', 'import', 'fit', 'profile')
                           if (example, step) != ('Oats', 'profile')}
    for key, entry in entries.items():
        assert json.loads(json.dumps(handle(key.split(' ')[1], copy.deepcopy(entry['request'])))) == entry['response']
    assert record().encode('utf-8') == original


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

