import json
from pathlib import Path
from torchoed_studio.server import handle

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'studio/torchoed_studio/static/replay.json'


def record():
    entries = {}
    for example in ('ChemReact', 'Oats'):
        oats = example == 'Oats'
        factors = ([dict(name='Variety', levels=['Victory', 'Golden Rain', 'Marvellous']),
                    dict(name='nitro', levels=[0, .2, .4, .6])] if oats else
                   [dict(name='Time', low=80, high=90), dict(name='Temp', low=170, high=180)])
        design = (dict(kind='split_plot_full_factorial', blocks=6, whole_plot=['Variety'], subplot=['nitro'])
                  if oats else dict(kind='central_composite', alpha=2**.5, center=[3, 3]))
        request = dict(factors=factors, model='linear' if oats else 'quadratic', design=design)
        def save(path, body):
            # Snapshot before handle adds fit terms, just as a JSON request crosses HTTP.
            frozen = json.loads(json.dumps(body))
            response = handle('/api/' + path, json.loads(json.dumps(body)))
            entries[example + ' /api/' + path] = dict(request=frozen, response=response)
            return response
        state = dict(request, **save('generate', request))
        for path in ('evaluate', 'compare', 'export'):
            save(path, state)
        request = dict(csv=(ROOT / ('reports/w17-reml-check/Oats.csv' if oats else
                                   'reports/w17-lm-published/ChemReact.csv')).read_text(),
                       factors=factors, response='yield' if oats else 'Yield', block='Block',
                       whole_plot=['Block', 'Variety'] if oats else None)
        state = dict(factors=factors, model='interaction' if oats else 'quadratic', **save('import', request))
        save('fit', state)
        if not oats:
            save('profile', state)
    return json.dumps(entries, indent=1, sort_keys=True)


if __name__ == '__main__':
    OUTPUT.write_text(record(), encoding='utf-8', newline='\n')
