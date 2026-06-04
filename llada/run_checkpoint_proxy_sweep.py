from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from checkpoint_proxy_selection import (
    extract_proxy_metric,
    load_results_payload,
    materialize_best_checkpoint,
    parse_optimizer_step_from_checkpoint_path,
    write_proxy_ranking_summary,
)
from humaneval_subset import build_humaneval_subset_manifest
from mbpp_subset import build_mbpp_subset_manifest


@dataclass(frozen=True)
class ProxyTaskSpec:
    record_name: str
    eval_task_name: str
    metric_task_name: str
    num_fewshot: int
    limit: int
    subset_manifest_path: str | None = None
    subset_manifest_env_var: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HumanEval/MBPP proxy evaluations over saved boundary step checkpoints."
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--eval_root", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--llada_dir", type=str, default=None)
    parser.add_argument("--accelerate_bin", type=str, default="accelerate")
    parser.add_argument("--gen_length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--boundary_threshold", type=float, default=0.5)
    parser.add_argument("--boundary_window_ratio", type=float, default=0.25)
    parser.add_argument("--boundary_threshold_grid", type=str, default=None)
    parser.add_argument("--boundary_window_ratio_grid", type=str, default=None)
    parser.add_argument("--phase_entropy_gate", type=float, default=0.80)
    parser.add_argument("--transition_weight", type=float, default=0.00)
    parser.add_argument("--phase_entropy_gate_grid", type=str, default=None)
    parser.add_argument("--transition_weight_grid", type=str, default=None)
    parser.add_argument("--runtime_mode", type=str, default="boundary_only")
    parser.add_argument("--scheduler_variant", type=str, default=None)
    parser.add_argument("--candidate_block_lengths", type=str, default=None)
    parser.add_argument("--max_block_length", type=int, default=None)
    parser.add_argument("--carry_weight", type=float, default=None)
    parser.add_argument("--hybrid_deltas", type=str, default=None)
    parser.add_argument("--disable_phase_aware_transfer", action="store_true")
    parser.add_argument("--disable_boundary_guard", action="store_true")
    parser.add_argument("--syntax_aware_landing", action="store_true")
    parser.add_argument("--commit_reopen_tokens", type=int, default=8)
    parser.add_argument("--humaneval_limit", type=int, default=50)
    parser.add_argument("--mbpp_limit", type=int, default=0)
    parser.add_argument("--humaneval_num_fewshot", type=int, default=0)
    parser.add_argument("--mbpp_num_fewshot", type=int, default=3)
    parser.add_argument("--humaneval_subset_strategy", type=str, default="stratified")
    parser.add_argument("--humaneval_subset_seed", type=int, default=20260416)
    parser.add_argument("--humaneval_subset_size", type=int, default=50)
    parser.add_argument("--humaneval_monitor_strategy", type=str, default="uniform")
    parser.add_argument("--humaneval_monitor_seed", type=int, default=20260417)
    parser.add_argument("--humaneval_monitor_size", type=int, default=50)
    parser.add_argument("--mbpp_subset_strategy", type=str, default="equidistant")
    parser.add_argument("--mbpp_subset_seed", type=int, default=20260419)
    parser.add_argument("--mbpp_subset_size", type=int, default=50)
    parser.add_argument("--disable_humaneval_monitor", action="store_true")
    parser.add_argument("--disable_mbpp", action="store_true")
    parser.add_argument("--best_output_path", type=str, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def _parse_float_grid(raw_value: str | None) -> list[float]:
    if raw_value is None:
        return []
    values: list[float] = []
    for part in str(raw_value).split(","):
        text = part.strip()
        if not text:
            continue
        values.append(float(text))
    return values


def expand_runtime_grid(
    boundary_threshold_grid: str | None,
    boundary_window_ratio_grid: str | None,
    *,
    phase_entropy_gate_grid: str | None = None,
    transition_weight_grid: str | None = None,
    runtime_mode: str = "phase_conditioned",
    default_boundary_threshold: float | None = None,
    default_boundary_window_ratio: float | None = None,
    default_phase_entropy_gate: float | None = None,
    default_transition_weight: float | None = None,
) -> list[dict[str, object]]:
    thresholds = _parse_float_grid(boundary_threshold_grid)
    if not thresholds:
        thresholds = [float(default_boundary_threshold if default_boundary_threshold is not None else 0.5)]
    window_ratios = _parse_float_grid(boundary_window_ratio_grid)
    if not window_ratios:
        window_ratios = [float(default_boundary_window_ratio if default_boundary_window_ratio is not None else 0.25)]
    entropy_gates = _parse_float_grid(phase_entropy_gate_grid)
    if not entropy_gates:
        entropy_gates = [float(default_phase_entropy_gate if default_phase_entropy_gate is not None else 0.80)]
    transition_weights = _parse_float_grid(transition_weight_grid)
    if not transition_weights:
        transition_weights = [float(default_transition_weight if default_transition_weight is not None else 1.00)]

    combinations: list[dict[str, object]] = []
    for boundary_threshold in thresholds:
        for boundary_window_ratio in window_ratios:
            for phase_entropy_gate in entropy_gates:
                for transition_weight in transition_weights:
                    combinations.append(
                        {
                            "boundary_threshold": float(boundary_threshold),
                            "boundary_window_ratio": float(boundary_window_ratio),
                            "phase_entropy_gate": float(phase_entropy_gate),
                            "transition_weight": float(transition_weight),
                            "runtime_mode": str(runtime_mode),
                        }
                    )
    return combinations


def _format_runtime_value(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def _load_step_metrics(checkpoint_dir: str | Path) -> dict[int, dict[str, object]]:
    metrics_path = Path(checkpoint_dir) / "metrics.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        return {}

    indexed: dict[int, dict[str, object]] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        optimizer_step = entry.get("optimizer_step")
        if isinstance(optimizer_step, int):
            indexed[optimizer_step] = entry
    return indexed


def write_runtime_grid_summary(path: str | Path, records: list[dict[str, object]]) -> dict[str, object]:
    summary = write_proxy_ranking_summary(path, records)
    best_record = summary.get("best_record") or {}
    summary["summary_type"] = "runtime_grid"
    summary["best_runtime_params"] = {
        "boundary_threshold": best_record.get("boundary_threshold"),
        "boundary_window_ratio": best_record.get("boundary_window_ratio"),
        "phase_entropy_gate": best_record.get("phase_entropy_gate"),
        "transition_weight": best_record.get("transition_weight"),
        "runtime_mode": best_record.get("runtime_mode"),
    }
    output_path = Path(path)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def discover_step_checkpoints(checkpoint_dir: str | Path) -> list[Path]:
    checkpoint_root = Path(checkpoint_dir)
    checkpoints = list(checkpoint_root.glob("boundary_head_step_*.pt"))
    if checkpoints:
        return sorted(
            checkpoints,
            key=lambda path: parse_optimizer_step_from_checkpoint_path(path) or 10**12,
        )

    last_checkpoint = checkpoint_root / "boundary_head_last.pt"
    if last_checkpoint.exists():
        return [last_checkpoint]
    return []


def find_latest_results_json(output_dir: str | Path) -> Path | None:
    candidates = sorted(
        Path(output_dir).rglob("results_*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        return None
    return candidates[-1]


def build_model_args(
    args: argparse.Namespace,
    checkpoint_path: str | Path,
    *,
    output_dir: str | Path,
    boundary_threshold: float,
    boundary_window_ratio: float,
    phase_entropy_gate: float,
    transition_weight: float,
    runtime_mode: str,
) -> str:
    def _encode_model_arg_list(raw_value: str | None) -> str | None:
        if raw_value is None:
            return None
        return str(raw_value).replace(",", "|")

    parts = [
        f"model_path={args.model_path}",
        f"gen_length={args.gen_length}",
        f"steps={args.steps}",
        f"block_length={args.block_length}",
        f"threshold={args.threshold}",
        f"boundary_head_path={checkpoint_path}",
        f"boundary_threshold={boundary_threshold}",
        f"boundary_window_ratio={boundary_window_ratio}",
        f"phase_entropy_gate={phase_entropy_gate}",
        f"transition_weight={transition_weight}",
        f"runtime_mode={runtime_mode}",
        "use_cache=False",
        "show_speed=True",
        f"save_dir={Path(output_dir) / 'generations'}",
        f"trace_dir={Path(output_dir) / 'traces'}",
    ]
    if args.scheduler_variant:
        parts.append(f"scheduler_variant={args.scheduler_variant}")
    if args.candidate_block_lengths:
        parts.append(f"candidate_block_lengths={_encode_model_arg_list(args.candidate_block_lengths)}")
    if args.max_block_length is not None:
        parts.append(f"max_block_length={args.max_block_length}")
    if args.carry_weight is not None:
        parts.append(f"carry_weight={args.carry_weight}")
    if args.hybrid_deltas:
        parts.append(f"hybrid_deltas={_encode_model_arg_list(args.hybrid_deltas)}")
    parts.append(f"phase_aware_transfer={not args.disable_phase_aware_transfer}")
    parts.append(f"boundary_guard={not args.disable_boundary_guard}")
    parts.append(f"syntax_aware_landing={bool(args.syntax_aware_landing)}")
    parts.append(f"commit_reopen_tokens={int(args.commit_reopen_tokens)}")
    return ",".join(parts)


def _iter_jsonl_payloads(root: Path, pattern: str):
    for path in sorted(root.rglob(pattern)):
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                yield json.loads(line)


def _completion_parse_ok(text: object) -> bool:
    completion = str(text or "")
    if not completion.strip():
        return False
    try:
        ast.parse(completion)
        return True
    except SyntaxError:
        pass
    body_lines = completion.splitlines()
    wrapped_body = "\n".join(
        line if line.startswith((" ", "\t")) or not line.strip() else f"    {line}"
        for line in body_lines
    )
    try:
        ast.parse("def _candidate():\n" + (wrapped_body or "    pass"))
        return True
    except SyntaxError:
        return False


def summarize_runtime_artifacts(output_dir: str | Path, *, target_block_length: int) -> dict[str, float]:
    root = Path(output_dir)
    block_lengths: list[int] = []
    sample_nfe: dict[str, float] = {}
    for event in _iter_jsonl_payloads(root / "traces", "rank_*.jsonl"):
        block_length = event.get("selected_block_length")
        if isinstance(block_length, int):
            block_lengths.append(int(block_length))
        total_nfe = event.get("sample_total_nfe")
        sample_key = str(event.get("sample_id") or event.get("request_index") or len(sample_nfe))
        if isinstance(total_nfe, (int, float)):
            sample_nfe[sample_key] = float(total_nfe)

    completions = list(_iter_jsonl_payloads(root / "generations", "rank_*.jsonl"))
    parse_rate = -1.0
    if completions:
        parse_rate = sum(1 for completion in completions if _completion_parse_ok(completion)) / len(completions)

    avg_block_length = -1.0
    block_length_error = -1.0
    if block_lengths:
        avg_block_length = sum(block_lengths) / len(block_lengths)
        block_length_error = abs(avg_block_length - float(target_block_length))

    avg_nfe = -1.0
    if sample_nfe:
        avg_nfe = sum(sample_nfe.values()) / len(sample_nfe)

    return {
        "parse_rate": parse_rate,
        "avg_block_length": avg_block_length,
        "block_length_distribution_error": block_length_error,
        "avg_nfe": avg_nfe,
    }


def run_proxy_eval(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    spec: ProxyTaskSpec,
    llada_dir: Path,
    eval_root: Path,
    boundary_threshold: float,
    boundary_window_ratio: float,
    phase_entropy_gate: float,
    transition_weight: float,
    runtime_mode: str,
) -> dict[str, object]:
    optimizer_step = parse_optimizer_step_from_checkpoint_path(checkpoint_path) or 0
    output_dir = eval_root / (
        f"{spec.record_name}_step_{optimizer_step:06d}"
        f"_bt_{_format_runtime_value(boundary_threshold)}"
        f"_bwr_{_format_runtime_value(boundary_window_ratio)}"
        f"_peg_{_format_runtime_value(phase_entropy_gate)}"
        f"_tw_{_format_runtime_value(transition_weight)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = find_latest_results_json(output_dir) if args.skip_existing else None
    if result_path is None:
        command = [
            args.accelerate_bin,
            "launch",
            "--num_processes=1",
            "eval_llada_semantic.py",
            "--tasks",
            spec.eval_task_name,
            "--num_fewshot",
            str(spec.num_fewshot),
            "--confirm_run_unsafe_code",
            "--model",
            "llada_semantic",
            "--model_args",
            build_model_args(
                args,
                checkpoint_path,
                output_dir=output_dir,
                boundary_threshold=boundary_threshold,
                boundary_window_ratio=boundary_window_ratio,
                phase_entropy_gate=phase_entropy_gate,
                transition_weight=transition_weight,
                runtime_mode=runtime_mode,
            ),
            "--output_path",
            str(output_dir),
            "--log_samples",
        ]
        include_path = llada_dir / "lm_eval_tasks"
        if include_path.exists():
            command.extend(["--include_path", str(include_path)])
        if spec.limit > 0:
            command.extend(["--limit", str(spec.limit)])

        env = os.environ.copy()
        env.setdefault("HF_ALLOW_CODE_EVAL", "1")
        env.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "true")
        if spec.subset_manifest_path is not None and spec.subset_manifest_env_var is not None:
            env[spec.subset_manifest_env_var] = str(spec.subset_manifest_path)
        subprocess.run(
            command,
            cwd=llada_dir,
            env=env,
            check=True,
        )
        result_path = find_latest_results_json(output_dir)

    if result_path is None:
        raise FileNotFoundError(f"No results_*.json found under {output_dir}")

    payload = load_results_payload(result_path)
    score = extract_proxy_metric(payload, spec.metric_task_name)
    runtime_summary = summarize_runtime_artifacts(output_dir, target_block_length=args.block_length)
    return {
        "task_name": spec.eval_task_name,
        "record_name": spec.record_name,
        "score": score,
        "results_path": str(result_path),
        "output_dir": str(output_dir),
        "boundary_threshold": float(boundary_threshold),
        "boundary_window_ratio": float(boundary_window_ratio),
        "phase_entropy_gate": float(phase_entropy_gate),
        "transition_weight": float(transition_weight),
        "runtime_mode": str(runtime_mode),
        **runtime_summary,
    }


def build_task_specs(args: argparse.Namespace, eval_root: Path) -> list[ProxyTaskSpec]:
    specs: list[ProxyTaskSpec] = []

    subset_strategy = str(args.humaneval_subset_strategy).strip().lower()
    if subset_strategy == "prefix":
        specs.append(
            ProxyTaskSpec(
                record_name="humaneval",
                eval_task_name="humaneval",
                metric_task_name="humaneval",
                num_fewshot=args.humaneval_num_fewshot,
                limit=args.humaneval_limit,
            )
        )
    else:
        manifest_root = eval_root / "subset_manifests"
        screening_manifest_path = manifest_root / "humaneval_screening_manifest.json"
        build_humaneval_subset_manifest(
            output_path=screening_manifest_path,
            subset_label="screening",
            strategy=subset_strategy,
            sample_size=args.humaneval_subset_size,
            seed=args.humaneval_subset_seed,
        )
        specs.append(
            ProxyTaskSpec(
                record_name="humaneval_screening",
                eval_task_name="llada_humaneval_subset",
                metric_task_name="llada_humaneval_subset",
                num_fewshot=args.humaneval_num_fewshot,
                limit=0,
                subset_manifest_path=str(screening_manifest_path),
                subset_manifest_env_var="LLADA_HUMANEVAL_SUBSET_MANIFEST",
            )
        )
        if not args.disable_humaneval_monitor:
            monitor_manifest_path = manifest_root / "humaneval_monitor_manifest.json"
            build_humaneval_subset_manifest(
                output_path=monitor_manifest_path,
                subset_label="monitor",
                strategy=args.humaneval_monitor_strategy,
                sample_size=args.humaneval_monitor_size,
                seed=args.humaneval_monitor_seed,
            )
            specs.append(
                ProxyTaskSpec(
                    record_name="humaneval_monitor",
                    eval_task_name="llada_humaneval_subset",
                    metric_task_name="llada_humaneval_subset",
                    num_fewshot=args.humaneval_num_fewshot,
                    limit=0,
                    subset_manifest_path=str(monitor_manifest_path),
                    subset_manifest_env_var="LLADA_HUMANEVAL_SUBSET_MANIFEST",
                )
            )
    if getattr(args, "disable_mbpp", False):
        return specs

    mbpp_subset_strategy = str(args.mbpp_subset_strategy).strip().lower()
    if mbpp_subset_strategy == "prefix":
        specs.append(
            ProxyTaskSpec(
                record_name="mbpp",
                eval_task_name="mbpp",
                metric_task_name="mbpp",
                num_fewshot=args.mbpp_num_fewshot,
                limit=args.mbpp_limit,
            )
        )
    else:
        manifest_root = eval_root / "subset_manifests"
        mbpp_manifest_path = manifest_root / "mbpp_screening_manifest.json"
        build_mbpp_subset_manifest(
            output_path=mbpp_manifest_path,
            subset_label="screening",
            strategy=mbpp_subset_strategy,
            sample_size=args.mbpp_subset_size,
            seed=args.mbpp_subset_seed,
        )
        specs.append(
            ProxyTaskSpec(
                record_name="mbpp",
                eval_task_name="llada_mbpp_subset",
                metric_task_name="llada_mbpp_subset",
                num_fewshot=args.mbpp_num_fewshot,
                limit=0,
                subset_manifest_path=str(mbpp_manifest_path),
                subset_manifest_env_var="LLADA_MBPP_SUBSET_MANIFEST",
            )
        )
    return specs


def main() -> int:
    args = parse_args()
    llada_dir = Path(args.llada_dir) if args.llada_dir else Path(__file__).resolve().parent
    eval_root = Path(args.eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)
    runtime_grid = expand_runtime_grid(
        args.boundary_threshold_grid,
        args.boundary_window_ratio_grid,
        phase_entropy_gate_grid=args.phase_entropy_gate_grid,
        transition_weight_grid=args.transition_weight_grid,
        runtime_mode=args.runtime_mode,
        default_boundary_threshold=args.boundary_threshold,
        default_boundary_window_ratio=args.boundary_window_ratio,
        default_phase_entropy_gate=args.phase_entropy_gate,
        default_transition_weight=args.transition_weight,
    )
    use_runtime_grid_summary = bool(
        args.boundary_threshold_grid
        or args.boundary_window_ratio_grid
        or args.phase_entropy_gate_grid
        or args.transition_weight_grid
    )
    summary_path = eval_root / ("runtime_grid_summary.json" if use_runtime_grid_summary else "proxy_ranking_summary.json")
    best_output_path = Path(args.best_output_path) if args.best_output_path else Path(args.checkpoint_dir) / "boundary_head_best.pt"

    checkpoints = discover_step_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No boundary_head_step_*.pt checkpoints found in {args.checkpoint_dir}")

    task_specs = build_task_specs(args, eval_root)
    metrics_by_step = _load_step_metrics(args.checkpoint_dir)

    records: list[dict[str, object]] = []
    for checkpoint_path in checkpoints:
        optimizer_step = parse_optimizer_step_from_checkpoint_path(checkpoint_path)
        print(f"[proxy-sweep] evaluating checkpoint={checkpoint_path} optimizer_step={optimizer_step}")
        for runtime_params in runtime_grid:
            boundary_threshold = float(runtime_params["boundary_threshold"])
            boundary_window_ratio = float(runtime_params["boundary_window_ratio"])
            phase_entropy_gate = float(runtime_params["phase_entropy_gate"])
            transition_weight = float(runtime_params["transition_weight"])
            runtime_mode = str(runtime_params["runtime_mode"])
            per_task = {}
            for spec in task_specs:
                result = run_proxy_eval(
                    args=args,
                    checkpoint_path=checkpoint_path,
                    spec=spec,
                    llada_dir=llada_dir,
                    eval_root=eval_root,
                    boundary_threshold=boundary_threshold,
                    boundary_window_ratio=boundary_window_ratio,
                    phase_entropy_gate=phase_entropy_gate,
                    transition_weight=transition_weight,
                    runtime_mode=runtime_mode,
                )
                per_task[spec.record_name] = result
                print(
                    "[proxy-sweep] "
                    f"step={optimizer_step} bt={boundary_threshold:.2f} bwr={boundary_window_ratio:.2f} "
                    f"peg={phase_entropy_gate:.2f} tw={transition_weight:.2f} mode={runtime_mode} "
                    f"{spec.record_name} score={result['score']} result={result['results_path']}"
                )

            step_metrics = metrics_by_step.get(int(optimizer_step or 0), {})
            runtime_metric_source = per_task.get("humaneval_screening", per_task.get("humaneval", per_task.get("mbpp", {})))
            records.append(
                {
                    "checkpoint_path": str(checkpoint_path),
                    "optimizer_step": optimizer_step,
                    "boundary_threshold": float(boundary_threshold),
                    "boundary_window_ratio": float(boundary_window_ratio),
                    "phase_entropy_gate": float(phase_entropy_gate),
                    "transition_weight": float(transition_weight),
                    "runtime_mode": runtime_mode,
                    "humaneval_score": per_task.get("humaneval_screening", per_task.get("humaneval", {})).get("score"),
                    "screening_humaneval_score": per_task.get("humaneval_screening", {}).get("score"),
                    "monitor_humaneval_score": per_task.get("humaneval_monitor", {}).get("score"),
                    "mbpp_score": per_task.get("mbpp", {}).get("score"),
                    "parse_rate": runtime_metric_source.get("parse_rate"),
                    "avg_block_length": runtime_metric_source.get("avg_block_length"),
                    "block_length_distribution_error": runtime_metric_source.get("block_length_distribution_error"),
                    "avg_nfe": runtime_metric_source.get("avg_nfe"),
                    "valid_boundary_precision": step_metrics.get("valid_boundary_precision"),
                    "valid_transition_f1": step_metrics.get("valid_transition_f1"),
                    "valid_phase_macro_f1": step_metrics.get("valid_phase_macro_f1"),
                    "valid_boundary_f1": step_metrics.get("valid_boundary_f1"),
                    "boundary_positive_rate_ratio": step_metrics.get("valid_boundary_positive_rate_ratio"),
                    "humaneval_results_path": per_task.get("humaneval_screening", per_task.get("humaneval", {})).get(
                        "results_path"
                    ),
                    "screening_humaneval_results_path": per_task.get("humaneval_screening", {}).get("results_path"),
                    "monitor_humaneval_results_path": per_task.get("humaneval_monitor", {}).get("results_path"),
                    "mbpp_results_path": per_task.get("mbpp", {}).get("results_path"),
                }
            )
        summary = (
            write_runtime_grid_summary(summary_path, records)
            if use_runtime_grid_summary
            else write_proxy_ranking_summary(summary_path, records)
        )
        materialized_path = materialize_best_checkpoint(summary, best_output_path)
        print(f"[proxy-sweep] summary updated: {summary_path}")
        print(f"[proxy-sweep] current best: {summary.get('best_checkpoint_path')}")
        if materialized_path is not None:
            print(f"[proxy-sweep] materialized best alias: {materialized_path}")

    print(f"[proxy-sweep] completed. summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
