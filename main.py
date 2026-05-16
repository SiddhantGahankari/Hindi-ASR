import argparse
import subprocess
import sys
from pathlib import Path


def _run_script(script_name, extra_args):
    repo_root = Path(__file__).resolve().parent
    script_path = repo_root / "scripts" / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    cmd = [sys.executable, str(script_path)] + extra_args
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Hindi ASR entrypoint for data, training, and inference scripts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("data", help="Run data pipeline (scripts/data_download.py)")
    subparsers.add_parser("train", help="Run training (scripts/train.py)")
    subparsers.add_parser("infer", help="Run inference (scripts/inference.py)")

    args, extra_args = parser.parse_known_args()

    if args.command == "data":
        _run_script("data_download.py", extra_args)
    elif args.command == "train":
        _run_script("train.py", extra_args)
    elif args.command == "infer":
        _run_script("inference.py", extra_args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
