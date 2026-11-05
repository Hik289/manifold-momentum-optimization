"""Archived remote-cache entries must not reappear as current experiment outcomes."""
import importlib.util,io,json,tarfile,tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];spec=importlib.util.spec_from_file_location('sync_results',ROOT/'experiments/extension/sync_results.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);m.ROOT=root;cache=root/'archive/extension_remote/experiments/extension/results';cache.mkdir(parents=True);(cache/'stale.json').write_text('{"status":"failed"}');(root/'experiments/extension/results').mkdir(parents=True)
 stream=io.BytesIO()
 with tarfile.open(fileobj=stream,mode='w:gz') as tf:
  data=b'{"status":"complete"}';info=tarfile.TarInfo('experiments/extension/results/current.json');info.size=len(data);tf.addfile(info,io.BytesIO(data))
 stream.seek(0)
 class Process:
  stdout=stream
  def wait(self):return 0
 with patch.object(m.subprocess,'Popen',return_value=Process()):m.sync()
 assert json.loads((root/'experiments/extension/results/current.json').read_text())['status']=='complete'
 assert not (root/'experiments/extension/results/stale.json').exists()
print({'passed':True,'archived_cache_outcomes_not_reintroduced':True})
