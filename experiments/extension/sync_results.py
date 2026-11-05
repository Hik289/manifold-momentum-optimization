"""Optionally copy result JSON and logs from a user-configured SSH host.

Set BURESMOMENTUM_REMOTE_HOST and BURESMOMENTUM_REMOTE_ROOT in your own shell.
No hostnames, account names, keys, or remote paths are stored in this repository.
"""
import datetime
import io
import json
import os
import shlex
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REMOTE_SCRIPT = r'''
import sys, tarfile
from pathlib import Path
root = Path(sys.argv[1]).expanduser()
files = []
for folder in ('experiments/extension/results', 'experiments/extension/logs'):
    files.extend(p for p in (root / folder).rglob('*')
                 if p.is_file() and p.suffix in ('.json', '.log'))
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as out:
    for path in sorted(files):
        try:
            out.add(path, arcname=str(path.relative_to(root)), recursive=False)
        except FileNotFoundError:
            pass
'''


def sync():
    host = os.environ.get('BURESMOMENTUM_REMOTE_HOST')
    remote_root = os.environ.get('BURESMOMENTUM_REMOTE_ROOT')
    if not host or not remote_root:
        raise RuntimeError('Set BURESMOMENTUM_REMOTE_HOST and BURESMOMENTUM_REMOTE_ROOT')
    command = 'python3 -c ' + shlex.quote(REMOTE_SCRIPT) + ' ' + shlex.quote(remote_root)
    proc = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, command],
        capture_output=True, check=True, timeout=180,
    )
    archive_root = ROOT / 'archive/extension_remote'
    archive_root.mkdir(parents=True, exist_ok=True)
    received = []
    with tarfile.open(fileobj=io.BytesIO(proc.stdout), mode='r:gz') as archive:
        for member in archive:
            dest = (archive_root / member.name).resolve()
            if archive_root.resolve() not in dest.parents or not member.isfile():
                raise ValueError('Unexpected archive member')
            archive.extract(member, path=archive_root)
            if Path(member.name).parent == Path('experiments/extension/results') \
                    and member.name.endswith('.json'):
                received.append(archive_root / member.name)
    target = ROOT / 'experiments/extension/results'
    target.mkdir(parents=True, exist_ok=True)
    for path in received:
        data = path.read_bytes()
        json.loads(data)
        temp = (target / path.name).with_suffix('.sync.tmp')
        temp.write_bytes(data)
        temp.replace(target / path.name)
    state = ROOT / 'analysis/REMOTE_SYNC_STATE.json'
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({
        'status': 'success',
        'last_successful_sync': datetime.datetime.now().astimezone().isoformat(),
        'result_files': len(received),
    }, indent=2) + '\n')


if __name__ == '__main__':
    sync()
