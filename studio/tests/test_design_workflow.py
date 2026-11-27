import csv, io, json
from collections import Counter
from pathlib import Path
from threading import Thread
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer
from torchoed import central_composite, full_factorial
from torchoed_studio.server import Handler, handle

ROOT = Path(__file__).resolve().parents[2]
E1 = dict(factors=[dict(name='Time',low=80,high=90),dict(name='Temp',low=170,high=180)],model='quadratic',design=dict(kind='central_composite',alpha=2**.5,center=[3,3]))
E2 = dict(factors=[dict(name='Variety',levels=['Victory','Golden Rain','Marvellous']),dict(name='nitro',levels=[0,.2,.4,.6])],model='linear',design=dict(kind='split_plot_full_factorial',blocks=6,whole_plot=['Variety'],subplot=['nitro']))


def test_chemreact():
    out = handle('/api/generate', E1)
    with (ROOT/'reports/w17-lm-published/ChemReact.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert Counter((round(r[0],2),round(r[1],2),b) for r,b in zip(out['runs'],out['block'])) == Counter((float(r['Time']),float(r['Temp']),int(r['Block'][1:])) for r in rows)
    assert out['runs'] == [[85+x*5,175+y*5] for x,y in central_composite(2,alpha=2**.5,center=(3,3)).tolist()]
    assert out['terms'] == ['Intercept','Time','Temp','Time^2','Temp^2','Time:Temp']
    assert out['model'] == 'quadratic'


def test_oats():
    out = handle('/api/generate', E2)
    with (ROOT/'reports/w17-reml-check/Oats.csv').open() as f:
        rows = list(csv.DictReader(f))
    blocks = list(dict.fromkeys(r['Block'] for r in rows))
    assert Counter((b,*r) for b,r in zip(out['block'],out['runs'])) == Counter((blocks.index(r['Block'])+1,r['Variety'],float(r['nitro'])) for r in rows)
    assert out['runs'] == [[E2['factors'][0]['levels'][i],E2['factors'][1]['levels'][j]] for _ in range(6) for i,j in full_factorial([3,4]).tolist()]
    assert out['whole_plot'] == [i+1 for i in range(18) for _ in range(4)]
    assert out['terms'] == ['Intercept','Variety','nitro']


def test_export_and_http():
    for body in (E1,E2):
        out = handle('/api/generate',body)
        rows = list(csv.DictReader(io.StringIO(handle('/api/export',out))))
        assert [[r[n] if f['name']=='Variety' else float(r[n]) for n,f in zip(out['columns'],body['factors'])] for r in rows] == out['runs']
        assert [int(r['block']) for r in rows] == out['block']
        if out['whole_plot'] is not None:
            assert [int(r['whole_plot']) for r in rows] == out['whole_plot']
    with ThreadingHTTPServer(('127.0.0.1',0),Handler) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            req = Request(f'http://127.0.0.1:{server.server_port}/api/generate',data=json.dumps(E1).encode(),headers={'Content-Type':'application/json'})
            with urlopen(req) as response:
                assert json.load(response) == handle('/api/generate',E1)
        finally:
            server.shutdown()
            thread.join()
