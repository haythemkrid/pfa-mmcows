import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

from shared.utils.logger import setup_logger

LOGGER = setup_logger(name="mmcows.main", log_file="logs/main.log")


def _parse_experiments(raw: str) -> List[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names:
        raise ValueError("No experiment names provided.")
    return names


def _normalize_experiment_record(record: object) -> Tuple[str, List[str]]:
    if isinstance(record, str):
        return record, []

    if not isinstance(record, dict):
        raise ValueError("Each experiment must be either a name string or an object.")

    name = str(record.get("name", "")).strip()
    if not name:
        raise ValueError("Experiment record is missing a non-empty 'name'.")

    overrides_raw = record.get("overrides", [])
    if isinstance(overrides_raw, dict):
        overrides = [f"{key}={value}" for key, value in overrides_raw.items()]
    elif isinstance(overrides_raw, list):
        overrides = [str(item) for item in overrides_raw]
    else:
        raise ValueError("'overrides' must be either a dict or a list.")

    return name, overrides


def _load_experiment_file(path: str) -> List[Tuple[str, List[str]]]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Experiments file not found: {path}")

    text = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for YAML experiment files.") from exc
        payload = yaml.safe_load(text)

    if isinstance(payload, dict):
        experiments = payload.get("experiments", [])
    elif isinstance(payload, list):
        experiments = payload
    else:
        raise ValueError("Experiment file must contain a list or an object with 'experiments'.")

    return [_normalize_experiment_record(item) for item in experiments]


def _run_multimodal_experiment(run_name: Optional[str], overrides: List[str]) -> int:
    command = [
        sys.executable,
        "-m",
        "src.multimodal.pipelines.training_pipeline",
    ]

    if run_name:
        command.append(f"experiment.run_name={run_name}")

    command.extend(overrides)

    LOGGER.info("Running experiment '%s'", run_name if run_name else "<auto>")
    LOGGER.info("Command: %s", " ".join(command))
    completed = subprocess.run(command, check=False)
    return completed.returncode


def _run_batch(
    experiments: List[Tuple[str, List[str]]],
    base_overrides: List[str],
    stop_on_error: bool,
) -> int:
    failures: Dict[str, int] = {}

    for run_name, experiment_overrides in experiments:
        final_overrides = [*base_overrides, *experiment_overrides]
        code = _run_multimodal_experiment(run_name, final_overrides)
        if code != 0:
            failures[run_name] = code
            LOGGER.error("Experiment '%s' failed with code %d", run_name, code)
            if stop_on_error:
                break

    if failures:
        for name, code in failures.items():
            LOGGER.error("FAILED: %s (exit code=%d)", name, code)
        return 1

    LOGGER.info("All experiments completed successfully.")
    return 0


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="MMCows multimodal experiment orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "multimodal",
        help="Run one multimodal experiment with optional Hydra overrides",
    )
    run_parser.add_argument(
        "--run-name",
        required=False,
        help="Optional Hydra run name (experiment.run_name). If omitted, config default is used.",
    )
    run_parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Hydra overrides, e.g. training.lr=3e-4 data.split_type=s2",
    )

    batch_parser = subparsers.add_parser(
        "multimodal-batch",
        help="Run multiple multimodal experiments",
    )
    batch_source = batch_parser.add_mutually_exclusive_group(required=True)
    batch_source.add_argument(
        "--experiments",
        help="Comma-separated run names, e.g. baseline,lr_sweep,s2_split",
    )
    batch_source.add_argument(
        "--experiments-file",
        help="Path to JSON/YAML list of experiments",
    )
    batch_parser.add_argument(
        "--base-overrides",
        nargs="*",
        default=[],
        help="Hydra overrides applied to every experiment",
    )
    batch_parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop as soon as one experiment fails",
    )

    args = parser.parse_args()

    if args.command == "multimodal":
        code = _run_multimodal_experiment(args.run_name, args.overrides)
        sys.exit(code)

    if args.experiments_file:
        experiments = _load_experiment_file(args.experiments_file)
    else:
        experiments = [(name, []) for name in _parse_experiments(args.experiments)]

    code = _run_batch(
        experiments=experiments,
        base_overrides=args.base_overrides,
        stop_on_error=args.stop_on_error,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
