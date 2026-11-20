"""Record paper-reference access outcomes; no design is reconstructed."""
import json
from pathlib import Path

results = {
    "B1": {
        "source": "DuMouchel and Jones (1994), Technometrics 36(1), pp. 37–47",
        "url": "https://doi.org/10.1080/00401706.1994.10485399",
        "verdict": "NR",
        "reason": "Publisher abstract accessible; full-text epdf endpoint inaccessible via web tool. No verbatim design table, table number/page, primary/potential terms and prior scale verified.",
    },
    "B2": {
        "source": "Li and Nachtsheim (2000), Technometrics 42(4), pp. 345–352",
        "url": "https://doi.org/10.1080/00401706.2000.10485707",
        "verdict": "NR",
        "reason": "Publisher abstract accessible; full-text epdf endpoint inaccessible via web tool. No verbatim design table, table number/page and complete model set verified.",
    },
}
for row in results.values():
    row.update(published=None, torchoed=None, delta=None)
results["published_example_count"] = 16
Path(__file__).with_name("results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(results, indent=2, ensure_ascii=False))
