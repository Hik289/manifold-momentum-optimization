"""Run one declared architecture-expansion job by ID from the repository root."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--suite-data", type=Path, default=ROOT / "data/expansion")
    args = parser.parse_args()
    jobs = json.loads((ROOT / "experiments/expansion/jobs.json").read_text())
    matches = [job for job in jobs if job["id"] == args.job_id]
    if len(matches) != 1:
        parser.error(f"Unknown job ID: {args.job_id}")
    job_file = ROOT / "experiments/expansion/job_files" / (args.job_id + ".json")
    job_file.parent.mkdir(parents=True, exist_ok=True)
    job_file.write_text(json.dumps(matches[0], indent=2) + "\n")
    command = [sys.executable, str(ROOT / "experiments/expansion/run_benchmark.py"),
               "--job", str(job_file), "--device", args.device,
               "--data", str(args.data), "--suite-data", str(args.suite_data)]
    raise SystemExit(subprocess.call(command, cwd=ROOT))


if __name__ == "__main__":
    main()
