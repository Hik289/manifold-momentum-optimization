"""Check that every plotted scalar is traceable to an actual saved observation."""
import json,math
from pathlib import Path
root=Path(__file__).resolve().parents[2];manifest=json.loads((root/'analysis/LOSS_FIGURE_PROVENANCE.json').read_text());seen=set();points=0
for figure,curves in manifest.items():
    assert (root/'paper/figures'/f'{figure}.pdf').stat().st_size>0
    for curve in curves:
        r=json.loads((root/'experiments/extension/results'/f"{curve['run']}.json").read_text());field=curve['field']
        rows=[v for v in r['history'] if isinstance(v.get(field),(int,float)) and math.isfinite(v[field]) and v[field]>=0]
        assert len(rows)==curve['points'],(figure,curve['run'],'observation count')
        assert rows[-1]['epoch']==curve['last'],(figure,curve['run'],'actual endpoint')
        assert r['status']==curve['status'],(figure,curve['run'],'status label')
        seen.add(curve['run']);points+=curve['points']
results=dict(passed=True,figure_groups=len(manifest),distinct_run_records=len(seen),displayed_observations_including_repeated_references=points)
(root/'analysis/LOSS_PROVENANCE_TEST.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results))
