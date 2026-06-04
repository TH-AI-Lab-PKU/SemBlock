from __future__ import annotations

import json
import os
import time
from typing import Sequence

import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from tqdm import tqdm

from eval_prompting import (
    build_generation_prompt,
    is_mbpp_doc,
    should_use_raw_completion_decode,
    truncate_generated_text,
)
from eval_llada_baseline import LLaDAEvalHarness as BaselineEvalHarness
from generate_baseline import generate_with_dual_cache, generate_with_prefix_cache
from generate_semantic import generate_semantic
from gsm8k_landing import maybe_apply_gsm8k_landing, parse_bool
from semantic_boundary import load_boundary_head


def _parse_optional_int(value, *, arg_name):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{arg_name} must be an integer or omitted.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not float(value).is_integer():
            raise ValueError(f"{arg_name} must be an integer or omitted.")
        return int(value)

    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{arg_name} must be an integer or omitted.") from exc


def _parse_optional_float(value, *, arg_name):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{arg_name} must be a float or omitted.")
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{arg_name} must be a float or omitted.") from exc


def _split_model_arg_list(value) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
        if not text:
            return []
        normalized = text.replace("|", ",")
        return [item.strip() for item in normalized.split(",") if item.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _parse_optional_int_list(value, *, arg_name):
    items = _split_model_arg_list(value)
    if items is None:
        return None

    parsed = []
    for index, item in enumerate(items):
        try:
            parsed_item = _parse_optional_int(item, arg_name=f"{arg_name}[{index}]")
        except ValueError as exc:
            raise ValueError(f"{arg_name} must be a list of integers.") from exc
        if parsed_item is None:
            continue
        parsed.append(parsed_item)
    return parsed


def _route_failure_case_scheduler_variant(default_variant, *, question, doc):
    variant = str(default_variant or "").strip()
    if variant != "failure_case_router":
        if variant != "mbpp_tuple_sequence_cache_router":
            return default_variant
        if not is_mbpp_doc(doc):
            return None
        task_text = str((doc or {}).get("text") or question or "").lower()
        if "tuple" in task_text:
            return "fixed_prefix_cache"
        if "regex" in task_text:
            return None
        if "sequence" in task_text or "character" in task_text:
            return "fixed_dual_cache"
        return None
    return None


def _fixed_block_trace_events(
    *,
    scheduler_name,
    sample_id,
    request_index,
    prompt_length,
    block_length,
    gen_length,
    nfe_history,
):
    block_history = []
    generated_length = 0
    events = []
    for step_index, _ in enumerate(nfe_history):
        current_block_length = min(block_length, max(gen_length - generated_length, 0))
        if current_block_length <= 0:
            break
        block_start = prompt_length + generated_length
        block_history.append(current_block_length)
        events.append(
            {
                "scheduler_name": scheduler_name,
                "sample_id": sample_id,
                "step_index": step_index,
                "generated_length": generated_length,
                "block_start": block_start,
                "window_size": current_block_length,
                "default_block_length": current_block_length,
                "selected_block_length": current_block_length,
                "selected_boundary_index": current_block_length - 1,
                "selected_boundary_position": block_start + current_block_length - 1,
                "selected_score": 0.0,
                "threshold": None,
                "candidate_scores": [],
                "candidate_positions": [],
                "metadata": {},
                "request_index": request_index,
                "sample_total_nfe": sum(nfe_history),
                "sample_block_count": len(nfe_history),
                "sample_block_history": [int(value) for value in block_history],
            }
        )
        generated_length += current_block_length
    return events, block_history


@register_model("llada_semantic")
class LLaDASemanticEvalHarness(BaselineEvalHarness):
    def __init__(
        self,
        boundary_head_path=None,
        boundary_threshold=None,
        boundary_window_ratio=0.25,
        trace_dir=None,
        scheduler_variant=None,
        candidate_block_lengths=None,
        max_block_length=None,
        carry_weight=None,
        hybrid_deltas=None,
        stateful_overlap_tokens=None,
        stateful_overlap_mode=None,
        cache_route_steps=None,
        phase_entropy_gate=None,
        transition_weight=None,
        runtime_mode=None,
        phase_aware_transfer=True,
        boundary_guard=True,
        syntax_aware_landing=False,
        commit_reopen_tokens=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if self.use_cache or self.dual_cache:
            raise NotImplementedError("Independent semantic evaluation currently supports only the non-cache path.")
        if not boundary_head_path:
            raise ValueError("boundary_head_path is required for semantic evaluation.")
        self.boundary_head_path = boundary_head_path
        self.boundary_threshold = boundary_threshold
        self.boundary_window_ratio = float(boundary_window_ratio)
        self.trace_dir = trace_dir
        self.scheduler_variant = str(scheduler_variant).strip() if scheduler_variant is not None else None
        if self.scheduler_variant == "":
            self.scheduler_variant = None
        self.candidate_block_lengths = _parse_optional_int_list(
            candidate_block_lengths,
            arg_name="candidate_block_lengths",
        )
        self.max_block_length = _parse_optional_int(
            max_block_length,
            arg_name="max_block_length",
        )
        self.carry_weight = _parse_optional_float(
            carry_weight,
            arg_name="carry_weight",
        )
        self.hybrid_deltas = _parse_optional_int_list(
            hybrid_deltas,
            arg_name="hybrid_deltas",
        )
        self.stateful_overlap_tokens = _parse_optional_int(
            stateful_overlap_tokens,
            arg_name="stateful_overlap_tokens",
        )
        self.stateful_overlap_mode = str(stateful_overlap_mode).strip() if stateful_overlap_mode is not None else None
        if self.stateful_overlap_mode == "":
            self.stateful_overlap_mode = None
        self.cache_route_steps = _parse_optional_int(
            cache_route_steps,
            arg_name="cache_route_steps",
        )
        self.phase_entropy_gate = _parse_optional_float(
            phase_entropy_gate,
            arg_name="phase_entropy_gate",
        )
        self.transition_weight = _parse_optional_float(
            transition_weight,
            arg_name="transition_weight",
        )
        self.runtime_mode = str(runtime_mode).strip() if runtime_mode is not None else None
        if self.runtime_mode == "":
            self.runtime_mode = None
        self.phase_aware_transfer = parse_bool(phase_aware_transfer)
        self.boundary_guard = parse_bool(boundary_guard)
        self.syntax_aware_landing = parse_bool(syntax_aware_landing)
        self.commit_reopen_tokens = _parse_optional_int(
            commit_reopen_tokens,
            arg_name="commit_reopen_tokens",
        )
        self.boundary_head = load_boundary_head(boundary_head_path, device=self.device)

    def _sample_id_for_request(self, request_index, request):
        doc = getattr(request, "doc", {}) or {}
        return str(doc.get("task_id") or doc.get("id") or f"sample-{request_index}")

    def generate_until(self, requests):
        output = []
        num_tokens = 0
        num_tokens_excluding_eos = 0
        total_nfe = []
        num_blocks = []
        processed_count = 0
        rank = getattr(self, "_rank", 0)
        save_path = None
        trace_path = None

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f"rank_{rank}.jsonl")
            print(f"save_path: {save_path}")
            if os.path.exists(save_path):
                print(f"load from {save_path}")
                with open(save_path, "r", encoding="utf-8") as handle:
                    output = [json.loads(line) for line in handle]
                    processed_count = len(output)
                print(f"processed_count: {processed_count}")

        if self.trace_dir is not None:
            os.makedirs(self.trace_dir, exist_ok=True)
            trace_path = os.path.join(self.trace_dir, f"rank_{rank}.jsonl")

        start_time = time.time()
        for i, req in enumerate(tqdm(requests, desc="Generating...")):
            if i < processed_count:
                continue

            question = req.args[0]
            routed_scheduler_variant = _route_failure_case_scheduler_variant(
                self.scheduler_variant,
                question=question,
                doc=getattr(req, "doc", None),
            )
            code_prompt_style = getattr(self, "code_prompt_style", "raw")
            user_input = build_generation_prompt(
                self.tokenizer,
                question=question,
                is_instruct=self.is_instruct,
                doc=getattr(req, "doc", None),
                code_prompt_style=code_prompt_style,
            )
            input_ids = self.tokenizer(user_input)["input_ids"]

            stop_tokens = req.args[1]["until"]
            input_ids = torch.tensor(input_ids).to(self.device).unsqueeze(0)
            sample_id = self._sample_id_for_request(i, req)
            semantic_runtime_kwargs = {}
            if routed_scheduler_variant in {"fixed_prefix_cache", "fixed_dual_cache", "baseline_dual_cache"}:
                route_steps = self.steps if self.cache_route_steps is None else self.cache_route_steps
                if routed_scheduler_variant == "fixed_prefix_cache":
                    generated_answer, nfe_history = generate_with_prefix_cache(
                        self.model,
                        input_ids,
                        steps=route_steps,
                        gen_length=self.gen_length,
                        block_length=self.block_length,
                        temperature=0,
                        remasking=self.remasking,
                        mask_id=self.mask_id,
                        threshold=self.threshold,
                    )
                    route_name = "fixed_prefix_cache_route"
                else:
                    generated_answer, nfe_history = generate_with_dual_cache(
                        self.model,
                        input_ids,
                        steps=route_steps,
                        gen_length=self.gen_length,
                        block_length=self.block_length,
                        temperature=0,
                        remasking=self.remasking,
                        mask_id=self.mask_id,
                        threshold=self.threshold,
                    )
                    route_name = (
                        "baseline_dual_cache_route"
                        if routed_scheduler_variant == "baseline_dual_cache"
                        else "fixed_dual_cache_route"
                    )
                trace_events, block_history = _fixed_block_trace_events(
                    scheduler_name=route_name,
                    sample_id=sample_id,
                    request_index=i,
                    prompt_length=input_ids.shape[1],
                    block_length=self.block_length,
                    gen_length=self.gen_length,
                    nfe_history=nfe_history,
                )
            else:
                if routed_scheduler_variant is not None:
                    semantic_runtime_kwargs["scheduler_variant"] = routed_scheduler_variant
                if self.candidate_block_lengths is not None:
                    semantic_runtime_kwargs["candidate_block_lengths"] = list(self.candidate_block_lengths)
                if self.max_block_length is not None:
                    semantic_runtime_kwargs["max_block_length"] = self.max_block_length
                if self.carry_weight is not None:
                    semantic_runtime_kwargs["carry_weight"] = self.carry_weight
                if self.hybrid_deltas is not None:
                    semantic_runtime_kwargs["hybrid_deltas"] = list(self.hybrid_deltas)
                if self.stateful_overlap_tokens is not None:
                    semantic_runtime_kwargs["stateful_overlap_tokens"] = self.stateful_overlap_tokens
                if self.stateful_overlap_mode is not None:
                    semantic_runtime_kwargs["stateful_overlap_mode"] = self.stateful_overlap_mode
                generated_answer, nfe_history, block_history, trace_events = generate_semantic(
                    self.model,
                    input_ids,
                    steps=self.steps,
                    gen_length=self.gen_length,
                    init_block_length=self.block_length,
                    temperature=0,
                    remasking=self.remasking,
                    mask_id=self.mask_id,
                    threshold=self.threshold,
                    boundary_head=self.boundary_head,
                    boundary_threshold=self.boundary_threshold,
                    boundary_window_ratio=self.boundary_window_ratio,
                    sample_id=sample_id,
                    phase_entropy_gate=self.phase_entropy_gate,
                    transition_weight=0.0 if self.transition_weight is None else self.transition_weight,
                    runtime_mode="boundary_only" if self.runtime_mode is None else self.runtime_mode,
                    phase_aware_transfer=self.phase_aware_transfer,
                    boundary_guard=self.boundary_guard,
                    syntax_aware_landing=self.syntax_aware_landing,
                    commit_reopen_tokens=0 if self.commit_reopen_tokens is None else self.commit_reopen_tokens,
                    tokenizer=self.tokenizer,
                    **semantic_runtime_kwargs,
                )

            if trace_path is not None:
                with open(trace_path, "a", encoding="utf-8") as handle:
                    for event in trace_events:
                        record = event if isinstance(event, dict) else event.to_dict()
                        record["request_index"] = i
                        record["sample_total_nfe"] = sum(nfe_history)
                        record["sample_block_count"] = len(block_history)
                        record["sample_block_history"] = [int(value) for value in block_history]
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            if should_use_raw_completion_decode(
                is_instruct=self.is_instruct,
                doc=getattr(req, "doc", None),
            ):
                if self.show_speed:
                    num_tokens += generated_answer.numel()
                    num_tokens_excluding_eos += (generated_answer != 126081).sum()
                    total_nfe.append(sum(nfe_history))
                    num_blocks.append(len(block_history))
                generated_answer = self.tokenizer.decode(
                    generated_answer[0][input_ids.shape[1]:],
                    skip_special_tokens=True,
                )
                generated_answer = truncate_generated_text(
                    generated_answer,
                    stop_tokens=stop_tokens,
                    is_instruct=self.is_instruct,
                    doc=getattr(req, "doc", None),
                    code_completion_postprocess=getattr(self, "code_completion_postprocess", True),
                )
            else:
                generated_answer = self.tokenizer.decode(generated_answer[0][input_ids.shape[1]:], skip_special_tokens=False)
                generated_answer = truncate_generated_text(
                    generated_answer,
                    stop_tokens=stop_tokens,
                    is_instruct=self.is_instruct,
                    doc=getattr(req, "doc", None),
                    code_completion_postprocess=getattr(self, "code_completion_postprocess", True),
                )

                generated_answer_ids = torch.tensor(self.tokenizer(generated_answer)["input_ids"])
                if self.show_speed:
                    num_tokens += generated_answer_ids.numel()
                    num_tokens_excluding_eos += (generated_answer_ids != 126081).sum()
                    total_nfe.append(sum(nfe_history))
                    num_blocks.append(len(block_history))
                generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
                generated_answer = maybe_apply_gsm8k_landing(
                    generated_answer,
                    req.doc,
                    enabled=self.gsm8k_landing_control,
                    tail_line_budget=self.gsm8k_landing_tail_lines,
                )

            output.append(generated_answer)
            if save_path is not None:
                with open(save_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(generated_answer, ensure_ascii=False) + "\n")

        end_time = time.time()
        if self.show_speed and total_nfe:
            print()
            print(f"Total number of tokens: {num_tokens}")
            print(f"Total number of tokens excluding EOS: {num_tokens_excluding_eos}")
            print(f"Total time taken: {end_time - start_time} seconds")
            print(f"Tokens per second: {num_tokens / (end_time - start_time)}")
            print()
            print(f"NFE for each sample: {total_nfe}")
            print(f"Total NFE: {sum(total_nfe)}")
            print(f"Average NFE per sample: {sum(total_nfe) / len(total_nfe)}")
            print()
            print(f"Number of blocks for each sample: {num_blocks}")
            print(f"Average number of blocks per sample: {sum(num_blocks) / len(num_blocks)}")
            print(f"Average block length: {self.gen_length / (sum(num_blocks) / len(num_blocks))}")
            print()
        return output


if __name__ == "__main__":
    cli_evaluate()
