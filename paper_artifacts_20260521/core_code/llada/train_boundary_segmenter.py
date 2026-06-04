from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from model.modeling_llada import LLaDAModelLM
except ImportError:  # pragma: no cover - allows lightweight unit imports without the model package.
    LLaDAModelLM = None

from evaluate_phase_boundary_labels import compute_label_metrics
from semantic_boundary import (
    BoundaryHeadConfig,
    SemanticBoundaryHead,
    build_boundary_labels_from_positions,
    build_boundary_labels_from_segments,
    infer_hidden_size,
    save_boundary_head,
)
from semantic_task_conditioned_head import SemanticTaskConditionedHead, TaskConditionedHeadConfig
from step_checkpointing import (
    build_step_checkpoint_path,
    parse_step_list,
    resolve_save_every_steps,
    resolve_training_epochs,
    should_save_step_checkpoint,
)


class BoundaryJsonlDataset(Dataset):
    """
    Supported JSONL schemas:

    1. {"segments": ["segment a", " segment b", ...], "source": "gum"}
    2. {"input_ids": [...], "boundary_labels": [...], "source": "proofnet"}
    3. {"input_ids": [...], "boundary_positions": [...], "source": "juice"}
    4. Domain-aware rows with:
       {
         "input_ids": [...],
         "phase_labels": [...],
         "boundary_labels": [...],
         "phase_loss_mask": [...],
         "boundary_loss_mask": [...],
         "source_domain": "codecontests",
         "source_view": "contest_rich",
         "condition_mask": 1,
         "label_confidence": "gold",
         "phase_label_vocab": [...]
       }
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        max_length: int,
        include_terminal_boundary: bool = True,
    ):
        self.examples: List[Dict] = []
        self.positive_labels = 0
        self.total_labels = 0
        self.phase_label_vocab: List[str] | None = None
        self.boundary_type_vocab: List[str] | None = None

        with open(path, "r", encoding="utf-8") as handle:
            for line_idx, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)
                example = self._normalize_record(
                    record=record,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    include_terminal_boundary=include_terminal_boundary,
                )
                if example is None:
                    continue
                self.examples.append(example)
                labels = example["boundary_labels"]
                boundary_loss_mask = example.get("boundary_loss_mask", [1] * len(labels))
                self.positive_labels += int(
                    sum(int(label) * int(mask) for label, mask in zip(labels, boundary_loss_mask))
                )
                self.total_labels += int(sum(int(mask) for mask in boundary_loss_mask))
                if example.get("phase_label_vocab") and self.phase_label_vocab is None:
                    self.phase_label_vocab = list(example["phase_label_vocab"])
                if example.get("boundary_type_vocab") and self.boundary_type_vocab is None:
                    self.boundary_type_vocab = list(example["boundary_type_vocab"])

        if not self.examples:
            raise ValueError(f"No valid training examples were loaded from {path}.")

    @staticmethod
    def _normalize_record(
        record: Dict,
        tokenizer,
        max_length: int,
        include_terminal_boundary: bool,
    ) -> Optional[Dict]:
        if "segments" in record:
            input_ids, boundary_labels = build_boundary_labels_from_segments(
                tokenizer=tokenizer,
                segments=record["segments"],
                include_terminal_boundary=include_terminal_boundary,
            )
            phase_labels = [0] * len(input_ids)
            phase_loss_mask = [0] * len(input_ids)
            boundary_loss_mask = [1] * len(input_ids)
            transition_loss_mask = [0] * len(input_ids)
            typed_transition_labels = [0] * len(input_ids)
            boundary_type_labels = [0] * len(input_ids)
            boundary_type_loss_mask = [0] * len(input_ids)
        elif "input_ids" in record and "boundary_labels" in record:
            input_ids = [int(x) for x in record["input_ids"]]
            boundary_labels = [int(x) for x in record["boundary_labels"]]
            phase_labels = [int(x) for x in record.get("phase_labels", [0] * len(input_ids))]
            phase_loss_mask = [int(x) for x in record.get("phase_loss_mask", [0] * len(input_ids))]
            boundary_loss_mask = [int(x) for x in record.get("boundary_loss_mask", [1] * len(input_ids))]
            transition_loss_mask = [int(x) for x in record.get("transition_loss_mask", phase_loss_mask)]
            typed_transition_labels = [
                int(x) for x in record.get("typed_transition_labels", [0] * len(input_ids))
            ]
            boundary_type_labels = [
                int(x) for x in record.get("boundary_type_labels", [0] * len(input_ids))
            ]
            boundary_type_loss_mask = [
                int(x)
                for x in record.get(
                    "boundary_type_loss_mask",
                    [1 if int(label) > 0 else 0 for label in boundary_type_labels],
                )
            ]
        elif "input_ids" in record and "boundary_positions" in record:
            input_ids = [int(x) for x in record["input_ids"]]
            boundary_labels = build_boundary_labels_from_positions(
                input_ids=input_ids,
                boundary_positions=record["boundary_positions"],
            )
            phase_labels = [0] * len(input_ids)
            phase_loss_mask = [0] * len(input_ids)
            boundary_loss_mask = [1] * len(input_ids)
            transition_loss_mask = [0] * len(input_ids)
            typed_transition_labels = [0] * len(input_ids)
            boundary_type_labels = [0] * len(input_ids)
            boundary_type_loss_mask = [0] * len(input_ids)
        else:
            raise ValueError(
                "Each JSONL row must contain either segments, input_ids+boundary_labels, or input_ids+boundary_positions."
            )

        if len(input_ids) != len(boundary_labels):
            raise ValueError("input_ids and boundary labels must have the same length.")
        if len(phase_labels) != len(input_ids):
            raise ValueError("input_ids and phase labels must have the same length.")
        if len(phase_loss_mask) != len(input_ids):
            raise ValueError("input_ids and phase loss mask must have the same length.")
        if len(boundary_loss_mask) != len(input_ids):
            raise ValueError("input_ids and boundary loss mask must have the same length.")
        if len(transition_loss_mask) != len(input_ids):
            raise ValueError("input_ids and transition loss mask must have the same length.")
        if len(typed_transition_labels) != len(input_ids):
            raise ValueError("input_ids and typed transition labels must have the same length.")
        if len(boundary_type_labels) != len(input_ids):
            raise ValueError("input_ids and boundary type labels must have the same length.")
        if len(boundary_type_loss_mask) != len(input_ids):
            raise ValueError("input_ids and boundary type loss mask must have the same length.")
        if not input_ids:
            return None

        input_ids = input_ids[:max_length]
        boundary_labels = boundary_labels[:max_length]
        phase_labels = phase_labels[:max_length]
        phase_loss_mask = phase_loss_mask[:max_length]
        boundary_loss_mask = boundary_loss_mask[:max_length]
        transition_loss_mask = transition_loss_mask[:max_length]
        typed_transition_labels = typed_transition_labels[:max_length]
        boundary_type_labels = boundary_type_labels[:max_length]
        boundary_type_loss_mask = boundary_type_loss_mask[:max_length]

        # If truncation cuts the sequence, we cannot trust that the truncated last token
        # still closes a semantic unit.
        if len(input_ids) == max_length:
            boundary_labels[-1] = 0
            typed_transition_labels[-1] = 0
            boundary_type_labels[-1] = 0

        return {
            "input_ids": input_ids,
            "boundary_labels": boundary_labels,
            "phase_labels": phase_labels,
            "phase_loss_mask": phase_loss_mask,
            "boundary_loss_mask": boundary_loss_mask,
            "transition_loss_mask": transition_loss_mask,
            "typed_transition_labels": typed_transition_labels,
            "boundary_type_labels": boundary_type_labels,
            "boundary_type_loss_mask": boundary_type_loss_mask,
            "loss_mask": boundary_loss_mask,
            "source": record.get("source", record.get("source_domain", "unknown")),
            "source_domain": record.get("source_domain", record.get("source", "unknown")),
            "source_view": record.get("source_view", "legacy"),
            "condition_mask": int(record.get("condition_mask", 0)),
            "public_test_token_span": [
                int(value) for value in record.get("public_test_token_span", [-1, -1])
            ],
            "label_confidence": record.get("label_confidence", "silver"),
            "hard_mining_cluster": record.get("hard_mining_cluster", "general"),
            "phase_label_vocab": record.get("phase_label_vocab"),
            "boundary_type_vocab": record.get("boundary_type_vocab"),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]
        return {
            "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
            "boundary_labels": torch.tensor(example["boundary_labels"], dtype=torch.float32),
            "phase_labels": torch.tensor(example["phase_labels"], dtype=torch.long),
            "phase_loss_mask": torch.tensor(example["phase_loss_mask"], dtype=torch.float32),
            "boundary_loss_mask": torch.tensor(example["boundary_loss_mask"], dtype=torch.float32),
            "transition_loss_mask": torch.tensor(example["transition_loss_mask"], dtype=torch.float32),
            "typed_transition_labels": torch.tensor(example["typed_transition_labels"], dtype=torch.long),
            "boundary_type_labels": torch.tensor(example["boundary_type_labels"], dtype=torch.long),
            "boundary_type_loss_mask": torch.tensor(example["boundary_type_loss_mask"], dtype=torch.float32),
            "loss_mask": torch.tensor(example["loss_mask"], dtype=torch.float32),
            "source": example["source"],
            "source_domain": example["source_domain"],
            "source_view": example["source_view"],
            "condition_mask": example["condition_mask"],
            "public_test_token_span": torch.tensor(example["public_test_token_span"], dtype=torch.long),
            "label_confidence": example["label_confidence"],
            "hard_mining_cluster": example["hard_mining_cluster"],
        }


class BoundaryCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        max_length = max(item["input_ids"].shape[0] for item in batch)

        input_ids = []
        attention_mask = []
        boundary_labels = []
        phase_labels = []
        loss_mask = []
        phase_loss_mask = []
        boundary_loss_mask = []
        transition_loss_mask = []
        typed_transition_labels = []
        boundary_type_labels = []
        boundary_type_loss_mask = []
        source_domain = []
        source_view = []
        condition_mask = []
        public_test_token_span = []
        label_confidence = []
        hard_mining_cluster = []

        for item in batch:
            seq_len = item["input_ids"].shape[0]
            pad_len = max_length - seq_len
            default_float_mask = torch.zeros(seq_len, dtype=torch.float32)
            default_long_labels = torch.zeros(seq_len, dtype=torch.long)
            item_transition_loss_mask = item.get("transition_loss_mask", item.get("phase_loss_mask", default_float_mask))
            item_typed_transition_labels = item.get("typed_transition_labels", default_long_labels)
            item_boundary_type_labels = item.get("boundary_type_labels", default_long_labels)
            item_boundary_type_loss_mask = item.get(
                "boundary_type_loss_mask",
                (item["boundary_labels"] > 0).to(torch.float32),
            )

            input_ids.append(
                F.pad(item["input_ids"], (0, pad_len), value=self.pad_token_id)
            )
            attention_mask.append(
                F.pad(torch.ones(seq_len, dtype=torch.long), (0, pad_len), value=0)
            )
            boundary_labels.append(
                F.pad(item["boundary_labels"], (0, pad_len), value=0)
            )
            phase_labels.append(
                F.pad(item["phase_labels"], (0, pad_len), value=0)
            )
            loss_mask.append(
                F.pad(item.get("loss_mask", torch.ones(seq_len, dtype=torch.float32)), (0, pad_len), value=0)
            )
            phase_loss_mask.append(
                F.pad(item["phase_loss_mask"], (0, pad_len), value=0)
            )
            boundary_loss_mask.append(
                F.pad(item["boundary_loss_mask"], (0, pad_len), value=0)
            )
            transition_loss_mask.append(
                F.pad(item_transition_loss_mask, (0, pad_len), value=0)
            )
            typed_transition_labels.append(
                F.pad(item_typed_transition_labels, (0, pad_len), value=0)
            )
            boundary_type_labels.append(
                F.pad(item_boundary_type_labels, (0, pad_len), value=0)
            )
            boundary_type_loss_mask.append(
                F.pad(item_boundary_type_loss_mask, (0, pad_len), value=0)
            )
            source_domain.append(item["source_domain"])
            source_view.append(item["source_view"])
            condition_mask.append(int(item["condition_mask"]))
            public_test_token_span.append(
                item.get("public_test_token_span", torch.tensor([-1, -1], dtype=torch.long))
            )
            label_confidence.append(item["label_confidence"])
            hard_mining_cluster.append(item.get("hard_mining_cluster", "general"))

        return {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "boundary_labels": torch.stack(boundary_labels, dim=0),
            "phase_labels": torch.stack(phase_labels, dim=0),
            "loss_mask": torch.stack(loss_mask, dim=0),
            "phase_loss_mask": torch.stack(phase_loss_mask, dim=0),
            "boundary_loss_mask": torch.stack(boundary_loss_mask, dim=0),
            "transition_loss_mask": torch.stack(transition_loss_mask, dim=0),
            "typed_transition_labels": torch.stack(typed_transition_labels, dim=0),
            "boundary_type_labels": torch.stack(boundary_type_labels, dim=0),
            "boundary_type_loss_mask": torch.stack(boundary_type_loss_mask, dim=0),
            "source_domain": source_domain,
            "source_view": source_view,
            "condition_mask": torch.tensor(condition_mask, dtype=torch.long),
            "public_test_token_span": torch.stack(public_test_token_span, dim=0),
            "label_confidence": label_confidence,
            "hard_mining_cluster": hard_mining_cluster,
        }


def resolve_curriculum_view_weights(progress_ratio: float) -> Dict[str, float]:
    if float(progress_ratio) < 0.1:
        return {
            "codesearchnet_code_bridge": 0.20,
            "codesearchnet_mbpp_bridge": 0.25,
            "contest_description_only": 0.05,
            "leetcode_code_bridge": 0.50,
        }
    if float(progress_ratio) < 0.4:
        return {
            "codesearchnet_code_bridge": 0.20,
            "codesearchnet_mbpp_bridge": 0.25,
            "contest_description_only": 0.05,
            "leetcode_code_bridge": 0.50,
        }
    return {
        "codesearchnet_code_bridge": 0.15,
        "codesearchnet_mbpp_bridge": 0.30,
        "contest_description_only": 0.05,
        "leetcode_code_bridge": 0.50,
    }


def apply_condition_dropout_to_batch(
    *,
    batch: Dict,
    dropout_probability: float,
    pad_token_id: int,
    rng: torch.Generator | None = None,
) -> Dict:
    if dropout_probability <= 0:
        return batch

    updated_batch = dict(batch)
    input_ids = batch["input_ids"].clone()
    attention_mask = batch["attention_mask"].clone()
    spans = batch.get("public_test_token_span")
    if spans is None:
        return batch

    rng = rng or torch.Generator()
    eligible_views = {"contest_rich"}
    for row_idx, source_view in enumerate(batch.get("source_view", [])):
        if source_view not in eligible_views:
            continue
        if int(batch["condition_mask"][row_idx].item()) != 1:
            continue
        if torch.rand((), generator=rng).item() >= float(dropout_probability):
            continue
        start = int(spans[row_idx][0].item())
        end = int(spans[row_idx][1].item())
        if start < 0 or end <= start:
            continue
        input_ids[row_idx, start:end] = int(pad_token_id)
        attention_mask[row_idx, start:end] = 0

    updated_batch["input_ids"] = input_ids
    updated_batch["attention_mask"] = attention_mask
    return updated_batch


def build_view_index(examples: List[Dict]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = {}
    for index, example in enumerate(examples):
        grouped.setdefault(str(example.get("source_view", "legacy")), []).append(index)
    return grouped


def build_hard_mining_index(examples: List[Dict]) -> List[int]:
    hard_clusters = {
        "return_shaping",
        "normalization_sorting",
        "state_update",
        "helper_definition",
        "signature_order",
    }
    return [
        index
        for index, example in enumerate(examples)
        if str(example.get("hard_mining_cluster", "general")) in hard_clusters
    ]


def sample_curriculum_batch(
    *,
    dataset: BoundaryJsonlDataset,
    collator: BoundaryCollator,
    batch_size: int,
    progress_ratio: float,
    rng: random.Random,
) -> Dict[str, torch.Tensor]:
    view_index = build_view_index(dataset.examples)
    hard_indices = build_hard_mining_index(dataset.examples)
    desired_weights = resolve_curriculum_view_weights(progress_ratio)
    available_weights = {
        view_name: weight
        for view_name, weight in desired_weights.items()
        if view_index.get(view_name)
    }
    if not available_weights:
        available_weights = {
            view_name: 1.0 / max(len(view_index), 1)
            for view_name in view_index
    }
    views = list(available_weights)
    weights = [float(available_weights[view_name]) for view_name in views]
    chosen_view = rng.choices(views, weights=weights, k=1)[0]
    hard_quota = max(0.0, float(batch_size) * 0.3)
    hard_take = int(math.floor(hard_quota))
    if rng.random() < (hard_quota - hard_take):
        hard_take += 1
    hard_take = min(len(hard_indices), int(batch_size), hard_take)
    chosen_indices = []
    for _ in range(hard_take):
        chosen_indices.append(rng.choice(hard_indices))
    while len(chosen_indices) < int(batch_size):
        chosen_indices.append(rng.choice(view_index[chosen_view]))
    return collator([dataset[index] for index in chosen_indices])


def sample_noisy_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_id: int,
    max_noise_ratio: float,
) -> torch.Tensor:
    device = input_ids.device
    batch_size, seq_len = input_ids.shape

    ratio = torch.rand(batch_size, 1, device=device) * max_noise_ratio
    random_mask = torch.rand(batch_size, seq_len, device=device) < ratio
    valid_mask = attention_mask.bool()
    noisy_mask = random_mask & valid_mask

    # Keep at least one visible token per sequence so the representation does not collapse.
    visible_counts = (valid_mask & ~noisy_mask).sum(dim=1)
    for row_idx in range(batch_size):
        if visible_counts[row_idx].item() == 0:
            valid_positions = torch.nonzero(valid_mask[row_idx], as_tuple=False).squeeze(-1)
            noisy_mask[row_idx, valid_positions[0]] = False

    noisy_inputs = input_ids.clone()
    noisy_inputs[noisy_mask] = mask_id
    return noisy_inputs


def masked_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    raw_loss = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none",
        pos_weight=pos_weight,
    )
    masked_loss = raw_loss * loss_mask
    denom = loss_mask.sum().clamp_min(1.0)
    return masked_loss.sum() / denom


def masked_asymmetric_focal_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma_pos: float = 1.0,
    gamma_neg: float = 4.0,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    positive_prob = probabilities.clamp(1e-6, 1.0 - 1e-6)
    negative_prob = (1.0 - probabilities).clamp(1e-6, 1.0 - 1e-6)
    positive_loss = -labels * float(pos_weight) * ((1.0 - positive_prob) ** float(gamma_pos)) * torch.log(positive_prob)
    negative_loss = -(1.0 - labels) * ((1.0 - negative_prob) ** float(gamma_neg)) * torch.log(negative_prob)
    masked_loss = (positive_loss + negative_loss) * loss_mask
    denom = loss_mask.sum().clamp_min(1.0)
    return masked_loss.sum() / denom


def masked_phase_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    raw_loss = F.cross_entropy(
        logits.transpose(1, 2),
        labels,
        reduction="none",
    )
    masked_loss = raw_loss * loss_mask
    denom = loss_mask.sum().clamp_min(1.0)
    return masked_loss.sum() / denom


def resolve_training_stage(progress_ratio: float) -> str:
    if float(progress_ratio) < 0.10:
        return "stage_a"
    if float(progress_ratio) < 0.40:
        return "stage_b"
    return "stage_c"


def resolve_phase_teacher_forcing_ratio(progress_ratio: float) -> float:
    progress = float(progress_ratio)
    if progress < 0.40:
        return 1.0
    if progress >= 0.70:
        return 0.0
    return max(0.0, 1.0 - ((progress - 0.40) / 0.30))


def derive_transition_targets(
    phase_labels: torch.Tensor,
    phase_loss_mask: torch.Tensor,
    boundary_loss_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    next_phase_labels = torch.cat([phase_labels[:, 1:], phase_labels[:, -1:]], dim=1)
    next_phase_mask = torch.cat([phase_loss_mask[:, 1:], torch.zeros_like(phase_loss_mask[:, -1:])], dim=1)
    transition_labels = (phase_labels != next_phase_labels).to(torch.float32)
    transition_mask = phase_loss_mask * next_phase_mask
    if boundary_loss_mask is not None:
        transition_mask = transition_mask * boundary_loss_mask
    return transition_labels, transition_mask


def build_gold_phase_posteriors(
    phase_labels: torch.Tensor,
    phase_loss_mask: torch.Tensor,
    num_labels: int,
) -> torch.Tensor:
    posteriors = F.one_hot(phase_labels.clamp(min=0), num_classes=int(num_labels)).to(torch.float32)
    return posteriors * phase_loss_mask.unsqueeze(-1)


def boundary_positive_rate_regularizer(
    *,
    logits: torch.Tensor,
    loss_mask: torch.Tensor,
    target_positive_rate: float,
) -> torch.Tensor:
    masked_probabilities = torch.sigmoid(logits) * loss_mask
    denom = loss_mask.sum().clamp_min(1.0)
    predicted_rate = masked_probabilities.sum() / denom
    target_rate = torch.as_tensor(float(target_positive_rate), dtype=predicted_rate.dtype, device=predicted_rate.device)
    return (predicted_rate - target_rate).pow(2)


def build_confidence_weights(confidence_labels: List[str], device: torch.device) -> torch.Tensor:
    weight_map = {
        "gold": 1.0,
        "silver": 0.5,
        "ignore": 0.0,
    }
    weights = [weight_map.get(str(label).lower(), 0.5) for label in confidence_labels]
    return torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(1)


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved: Dict = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def extract_checkpoint_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Checkpoint must be either a state_dict or a payload with a state_dict field.")


def load_resume_checkpoint(boundary_head, resume_path: Path) -> None:
    checkpoint = torch.load(resume_path, map_location=next(boundary_head.parameters()).device)
    state_dict = extract_checkpoint_state_dict(checkpoint)
    model_keys = set(boundary_head.state_dict())
    loaded_keys = model_keys.intersection(state_dict)
    if not loaded_keys:
        raise RuntimeError(
            f"Resume checkpoint did not match any boundary head parameters: {resume_path}"
        )
    incompatible = boundary_head.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "Loaded resume checkpoint with non-strict key match: "
            f"loaded_keys={len(loaded_keys)}, "
            f"missing_keys={len(incompatible.missing_keys)}, "
            f"unexpected_keys={len(incompatible.unexpected_keys)}"
        )


@torch.no_grad()
def evaluate(
    llada_model,
    boundary_head,
    dataloader: DataLoader,
    mask_id: int,
    max_noise_ratio: float,
    device: torch.device,
    phase_label_vocab: Optional[List[str]] = None,
    target_boundary_positive_rate: Optional[float] = None,
) -> Dict[str, float]:
    llada_model.eval()
    boundary_head.eval()

    total_loss = 0.0
    total_phase_loss = 0.0
    total_transition_loss = 0.0
    total_boundary_type_loss = 0.0
    total_boundary_loss = 0.0
    total_batches = 0
    phase_predictions = []
    phase_targets = []
    phase_masks = []
    transition_predictions = []
    transition_targets = []
    transition_masks = []
    boundary_predictions = []
    boundary_targets = []
    boundary_masks = []
    total_predicted_boundary_positive = 0.0
    total_gold_boundary_positive = 0.0
    total_boundary_mask = 0.0
    grouped_metrics_inputs: Dict[str, Dict[str, List[List[int]]]] = {}

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        batch = move_batch_to_device(batch, device)
        noisy_inputs = sample_noisy_inputs(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            mask_id=mask_id,
            max_noise_ratio=max_noise_ratio,
        )

        outputs = llada_model(
            input_ids=noisy_inputs,
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1].to(torch.float32)
        confidence_weights = build_confidence_weights(batch["label_confidence"], device=device)
        scaled_boundary_mask = batch["boundary_loss_mask"] * confidence_weights
        scaled_phase_mask = batch["phase_loss_mask"] * confidence_weights
        transition_labels, transition_loss_mask = derive_transition_targets(
            batch["phase_labels"],
            batch["phase_loss_mask"],
            batch["boundary_loss_mask"],
        )
        if hasattr(boundary_head, "predict_runtime_components"):
            components = boundary_head.predict_runtime_components(hidden_states)
            phase_logits = components["phase_logits"]
            transition_logits = components["transition_logits"]
            boundary_logits = components["boundary_logits"]
            typed_transition_logits = components.get("typed_transition_logits")
            boundary_type_logits = components.get("boundary_type_logits")
        else:
            boundary_logits = boundary_head(hidden_states)
            phase_logits = None
            transition_logits = None
            typed_transition_logits = None
            boundary_type_logits = None

        boundary_loss = masked_asymmetric_focal_loss(
            logits=boundary_logits,
            labels=batch["boundary_labels"],
            loss_mask=scaled_boundary_mask,
        )
        total_boundary_loss += float(boundary_loss.item())
        total_loss += float(boundary_loss.item())

        if phase_logits is not None:
            phase_loss = masked_phase_cross_entropy(
                logits=phase_logits,
                labels=batch["phase_labels"],
                loss_mask=scaled_phase_mask,
            )
            transition_loss = masked_asymmetric_focal_loss(
                logits=transition_logits,
                labels=transition_labels,
                loss_mask=transition_loss_mask * confidence_weights,
                gamma_pos=1.0,
                gamma_neg=2.0,
            )
            boundary_type_loss = torch.zeros((), device=device)
            if boundary_type_logits is not None:
                boundary_type_loss = masked_phase_cross_entropy(
                    logits=boundary_type_logits,
                    labels=batch["boundary_type_labels"],
                    loss_mask=batch["boundary_type_loss_mask"] * confidence_weights,
                )
            if typed_transition_logits is not None:
                boundary_type_loss = boundary_type_loss + masked_phase_cross_entropy(
                    logits=typed_transition_logits,
                    labels=batch["typed_transition_labels"],
                    loss_mask=batch.get("transition_loss_mask", transition_loss_mask) * confidence_weights,
                )
            total_phase_loss += float(phase_loss.item())
            total_transition_loss += float(transition_loss.item())
            total_boundary_type_loss += float(boundary_type_loss.item())
            total_loss += float(getattr(boundary_head.config, "phase_loss_weight", 1.0) * phase_loss.item())
            total_loss += float(getattr(boundary_head.config, "transition_loss_weight", 1.0) * transition_loss.item())
            total_loss += float(getattr(boundary_head.config, "boundary_type_loss_weight", 0.0) * boundary_type_loss.item())
            phase_predictions.extend(torch.argmax(phase_logits, dim=-1).detach().cpu().tolist())
            phase_targets.extend(batch["phase_labels"].detach().cpu().tolist())
            phase_masks.extend(batch["phase_loss_mask"].detach().cpu().int().tolist())
            transition_predictions.extend((torch.sigmoid(transition_logits) >= 0.5).detach().cpu().int().tolist())
            transition_targets.extend(transition_labels.detach().cpu().int().tolist())
            transition_masks.extend(transition_loss_mask.detach().cpu().int().tolist())
        else:
            phase_loss = torch.zeros((), device=device)
            transition_loss = torch.zeros((), device=device)
            boundary_type_loss = torch.zeros((), device=device)

        total_batches += 1

        batch_boundary_predictions = (torch.sigmoid(boundary_logits) >= 0.5).detach().cpu().int().tolist()
        batch_boundary_targets = batch["boundary_labels"].detach().cpu().int().tolist()
        batch_boundary_masks = batch["boundary_loss_mask"].detach().cpu().int().tolist()
        total_predicted_boundary_positive += float((torch.sigmoid(boundary_logits) * batch["boundary_loss_mask"]).sum().item())
        total_gold_boundary_positive += float((batch["boundary_labels"] * batch["boundary_loss_mask"]).sum().item())
        total_boundary_mask += float(batch["boundary_loss_mask"].sum().item())
        batch_phase_predictions = (
            torch.argmax(phase_logits, dim=-1).detach().cpu().tolist()
            if phase_logits is not None
            else [[0] * len(row) for row in batch_boundary_predictions]
        )
        batch_phase_targets = batch["phase_labels"].detach().cpu().tolist()
        batch_phase_masks = batch["phase_loss_mask"].detach().cpu().int().tolist()
        batch_transition_predictions = (
            (torch.sigmoid(transition_logits) >= 0.5).detach().cpu().int().tolist()
            if transition_logits is not None
            else [[0] * len(row) for row in batch_boundary_predictions]
        )
        batch_transition_targets = transition_labels.detach().cpu().int().tolist()
        batch_transition_masks = transition_loss_mask.detach().cpu().int().tolist()

        boundary_predictions.extend(batch_boundary_predictions)
        boundary_targets.extend(batch_boundary_targets)
        boundary_masks.extend(batch_boundary_masks)

        for row_idx, source_view in enumerate(batch.get("source_view", [])):
            group = grouped_metrics_inputs.setdefault(
                str(source_view),
                {
                    "phase_predictions": [],
                    "phase_targets": [],
                    "phase_masks": [],
                    "transition_predictions": [],
                    "transition_targets": [],
                    "transition_masks": [],
                    "boundary_predictions": [],
                    "boundary_targets": [],
                    "boundary_masks": [],
                },
            )
            group["phase_predictions"].append(batch_phase_predictions[row_idx])
            group["phase_targets"].append(batch_phase_targets[row_idx])
            group["phase_masks"].append(batch_phase_masks[row_idx])
            group["transition_predictions"].append(batch_transition_predictions[row_idx])
            group["transition_targets"].append(batch_transition_targets[row_idx])
            group["transition_masks"].append(batch_transition_masks[row_idx])
            group["boundary_predictions"].append(batch_boundary_predictions[row_idx])
            group["boundary_targets"].append(batch_boundary_targets[row_idx])
            group["boundary_masks"].append(batch_boundary_masks[row_idx])

    predicted_positive_rate = 0.0 if total_boundary_mask <= 0 else total_predicted_boundary_positive / total_boundary_mask
    gold_positive_rate = 0.0 if total_boundary_mask <= 0 else total_gold_boundary_positive / total_boundary_mask
    metrics = compute_label_metrics(
        phase_predictions=phase_predictions or [[0]],
        phase_targets=phase_targets or [[0]],
        phase_masks=phase_masks or [[0]],
        transition_predictions=transition_predictions or [[0]],
        transition_targets=transition_targets or [[0]],
        transition_masks=transition_masks or [[0]],
        boundary_predictions=boundary_predictions or [[0]],
        boundary_targets=boundary_targets or [[0]],
        boundary_masks=boundary_masks or [[0]],
        phase_label_vocab=phase_label_vocab or getattr(boundary_head.config, "phase_label_vocab", ["boundary_only"]),
    )
    metrics.update(
        {
            "loss": total_loss / max(total_batches, 1),
            "phase_loss": total_phase_loss / max(total_batches, 1),
            "transition_loss": total_transition_loss / max(total_batches, 1),
            "boundary_type_loss": total_boundary_type_loss / max(total_batches, 1),
            "boundary_loss": total_boundary_loss / max(total_batches, 1),
            "precision": metrics["boundary_precision"],
            "recall": metrics["boundary_recall"],
            "f1": metrics["boundary_f1"],
            "boundary_predicted_positive_rate": predicted_positive_rate,
            "boundary_gold_positive_rate": gold_positive_rate,
            "boundary_positive_rate_ratio": 0.0 if gold_positive_rate <= 0 else predicted_positive_rate / gold_positive_rate,
        }
    )
    if target_boundary_positive_rate is not None:
        metrics["boundary_positive_rate_target"] = float(target_boundary_positive_rate)
    for source_view, grouped in grouped_metrics_inputs.items():
        grouped_metrics = compute_label_metrics(
            phase_predictions=grouped["phase_predictions"] or [[0]],
            phase_targets=grouped["phase_targets"] or [[0]],
            phase_masks=grouped["phase_masks"] or [[0]],
            transition_predictions=grouped["transition_predictions"] or [[0]],
            transition_targets=grouped["transition_targets"] or [[0]],
            transition_masks=grouped["transition_masks"] or [[0]],
            boundary_predictions=grouped["boundary_predictions"] or [[0]],
            boundary_targets=grouped["boundary_targets"] or [[0]],
            boundary_masks=grouped["boundary_masks"] or [[0]],
            phase_label_vocab=phase_label_vocab or getattr(boundary_head.config, "phase_label_vocab", ["boundary_only"]),
        )
        metrics[f"by_view_{source_view}_phase_macro_f1"] = grouped_metrics["phase_macro_f1"]
        metrics[f"by_view_{source_view}_transition_f1"] = grouped_metrics["transition_f1"]
        metrics[f"by_view_{source_view}_boundary_f1"] = grouped_metrics["boundary_f1"]
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a semantic boundary head on top of frozen LLaDA hidden states."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--valid_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--projection_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_noise_ratio", type=float, default=0.5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--save_every_steps", type=int, default=None)
    parser.add_argument("--checkpoint_steps", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--condition_dropout_probability", type=float, default=0.5)
    parser.add_argument("--exclude_terminal_boundary", action="store_true")
    parser.add_argument("--boundary_pos_weight", type=float, default=None,
                        help="Multiplicative weight on positive (boundary) terms in focal loss. "
                             ">1.0 boosts recall; <1.0 boosts precision. Default: 1.0 (unchanged).")
    parser.add_argument("--phase_loss_weight", type=float, default=None,
                        help="Override the phase classification auxiliary loss weight.")
    parser.add_argument("--transition_loss_weight", type=float, default=None,
                        help="Override the binary phase-transition auxiliary loss weight.")
    parser.add_argument("--boundary_type_loss_weight", type=float, default=None,
                        help="Override the typed-boundary auxiliary loss weight.")
    parser.add_argument("--boundary_loss_weight", type=float, default=None,
                        help="Override the final safe-boundary loss weight.")
    parser.add_argument("--use_boundary_type_features", action="store_true",
                        help="Feed predicted boundary-type posteriors into the final boundary head. "
                             "Off by default because type labels are only supervised on positive boundaries.")
    parser.add_argument("--positive_rate_reg_weight", type=float, default=0.05,
                        help="Weight for matching the expected boundary positive rate.")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a .pt checkpoint to warm-start boundary head weights from.")
    return parser.parse_args()


def build_training_metadata(
    *,
    args: argparse.Namespace,
    history: List[Dict[str, float]],
    include_terminal_boundary: bool,
    optimizer_step: int,
    micro_step: int,
    selection_metric_source: str,
    save_every_steps: int | None,
    checkpoint_steps: List[int],
    target_boundary_positive_rate: float,
    positive_rate_source: str,
) -> Dict[str, object]:
    return {
        "model_path": args.model_path,
        "train_data": args.train_data,
        "valid_data": args.valid_data,
        "max_length": args.max_length,
        "max_noise_ratio": args.max_noise_ratio,
        "include_terminal_boundary": include_terminal_boundary,
        "history": history,
        "global_step": optimizer_step,
        "optimizer_step": optimizer_step,
        "micro_step": micro_step,
        "max_train_steps": args.max_train_steps,
        "save_every_steps": save_every_steps,
        "checkpoint_steps": checkpoint_steps,
        "recommended_checkpoint_selection": "proxy_eval",
        "selection_metric_source": selection_metric_source,
        "label_semantics": "boundary_after_token",
        "checkpoint_selection_disclosure": "Benchmark-driven proxy feedback is used for checkpoint choice; final scores are not strict hold-out selection.",
        "condition_dropout_probability": args.condition_dropout_probability,
        "phase_loss_weight": args.phase_loss_weight,
        "transition_loss_weight": args.transition_loss_weight,
        "boundary_type_loss_weight": args.boundary_type_loss_weight,
        "boundary_loss_weight": args.boundary_loss_weight,
        "use_boundary_type_features": bool(args.use_boundary_type_features),
        "positive_rate_reg_weight": args.positive_rate_reg_weight,
        "target_boundary_positive_rate": float(target_boundary_positive_rate),
        "positive_rate_source": positive_rate_source,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.epochs <= 0:
        raise ValueError("--epochs must be a positive integer.")
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        raise ValueError("--max_train_steps must be a positive integer when provided.")
    if args.save_every_steps is not None and args.save_every_steps <= 0:
        raise ValueError("--save_every_steps must be a positive integer when provided.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    include_terminal_boundary = not args.exclude_terminal_boundary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if LLaDAModelLM is None:
        raise ImportError("model.modeling_llada.LLaDAModelLM is required to run training.")

    llada_model = LLaDAModelLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    llada_model.eval()
    for param in llada_model.parameters():
        param.requires_grad = False

    train_dataset = BoundaryJsonlDataset(
        path=args.train_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        include_terminal_boundary=include_terminal_boundary,
    )
    valid_dataset = None
    if args.valid_data is not None:
        valid_dataset = BoundaryJsonlDataset(
            path=args.valid_data,
            tokenizer=tokenizer,
            max_length=args.max_length,
            include_terminal_boundary=include_terminal_boundary,
        )

    hidden_size = infer_hidden_size(llada_model)
    boundary_type_vocab = list(train_dataset.boundary_type_vocab or [])
    phase_loss_weight = float(args.phase_loss_weight) if args.phase_loss_weight is not None else 0.5
    transition_loss_weight = float(args.transition_loss_weight) if args.transition_loss_weight is not None else 0.75
    boundary_type_loss_weight = (
        float(args.boundary_type_loss_weight)
        if args.boundary_type_loss_weight is not None
        else 0.0
    )
    boundary_loss_weight = float(args.boundary_loss_weight) if args.boundary_loss_weight is not None else 1.0
    args.phase_loss_weight = phase_loss_weight
    args.transition_loss_weight = transition_loss_weight
    args.boundary_type_loss_weight = boundary_type_loss_weight
    args.boundary_loss_weight = boundary_loss_weight
    if train_dataset.phase_label_vocab:
        boundary_head = SemanticTaskConditionedHead(
            TaskConditionedHeadConfig(
                hidden_size=hidden_size,
                projection_size=args.projection_size,
                dropout=args.dropout,
                include_terminal_boundary=include_terminal_boundary,
                phase_label_vocab=list(train_dataset.phase_label_vocab),
                boundary_type_vocab=boundary_type_vocab,
                phase_loss_weight=phase_loss_weight,
                transition_loss_weight=transition_loss_weight,
                boundary_type_loss_weight=boundary_type_loss_weight,
                boundary_loss_weight=boundary_loss_weight,
                use_boundary_type_features=bool(args.use_boundary_type_features),
            )
        ).to(device)
    else:
        boundary_head = SemanticBoundaryHead(
            BoundaryHeadConfig(
                hidden_size=hidden_size,
                projection_size=args.projection_size,
                dropout=args.dropout,
                include_terminal_boundary=include_terminal_boundary,
            )
        ).to(device)

    if args.resume_from is not None:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume_from checkpoint not found: {resume_path}")
        load_resume_checkpoint(boundary_head, resume_path)

    boundary_pos_weight = float(args.boundary_pos_weight) if args.boundary_pos_weight is not None else 1.0

    collator = BoundaryCollator(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    valid_loader = None
    if valid_dataset is not None:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
        )

    target_boundary_positive_rate = 0.0
    positive_rate_source = "none"
    if train_dataset.total_labels > 0:
        target_boundary_positive_rate = train_dataset.positive_labels / max(train_dataset.total_labels, 1)
        positive_rate_source = "train"
    elif valid_dataset is not None and valid_dataset.total_labels > 0:
        target_boundary_positive_rate = valid_dataset.positive_labels / max(valid_dataset.total_labels, 1)
        positive_rate_source = "valid"

    optimizer = AdamW(
        boundary_head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    curriculum_enabled = any(
        str(example.get("source_view", "legacy")) != "legacy"
        for example in train_dataset.examples
    )
    curriculum_rng = random.Random(args.seed)
    dropout_rng = torch.Generator().manual_seed(args.seed)

    history: List[Dict[str, float]] = []
    optimizer_step = 0
    micro_step = 0
    mask_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else 126336
    configured_checkpoint_steps = parse_step_list(args.checkpoint_steps)
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    resolved_save_every_steps = resolve_save_every_steps(
        save_every_steps=args.save_every_steps,
        checkpoint_steps=configured_checkpoint_steps,
        max_train_steps=args.max_train_steps,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
    )
    effective_epochs = resolve_training_epochs(
        epochs=args.epochs,
        max_train_steps=args.max_train_steps,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
    )
    total_optimizer_steps_target = args.max_train_steps or (effective_epochs * optimizer_steps_per_epoch)
    saved_step_checkpoints: set[int] = set()
    stop_training = False
    step_selection_enabled = bool(
        args.max_train_steps is not None
        or args.save_every_steps is not None
        or configured_checkpoint_steps
    )
    selection_metric_source = "proxy_eval_required" if step_selection_enabled else "training_last_checkpoint"

    def save_current_step_checkpoint() -> None:
        if optimizer_step <= 0 or optimizer_step in saved_step_checkpoints:
            return
        save_boundary_head(
            boundary_head,
            build_step_checkpoint_path(output_dir, optimizer_step),
            metadata=build_training_metadata(
                args=args,
                history=history,
                include_terminal_boundary=include_terminal_boundary,
                optimizer_step=optimizer_step,
                micro_step=micro_step,
                selection_metric_source=selection_metric_source,
                save_every_steps=resolved_save_every_steps,
                checkpoint_steps=configured_checkpoint_steps,
                target_boundary_positive_rate=target_boundary_positive_rate,
                positive_rate_source=positive_rate_source,
            ),
        )
        saved_step_checkpoints.add(optimizer_step)

    if step_selection_enabled:
        print(
            "Step-based checkpoint selection enabled: "
            f"effective_epochs={effective_epochs}, max_train_steps={args.max_train_steps}, "
            f"save_every_steps={resolved_save_every_steps}, checkpoint_steps={configured_checkpoint_steps}"
        )

    for epoch in range(1, effective_epochs + 1):
        llada_model.eval()
        boundary_head.train()
        optimizer.zero_grad(set_to_none=True)

        progress = tqdm(range(1, len(train_loader) + 1), desc=f"Epoch {epoch}/{effective_epochs}")
        running_loss = 0.0
        running_phase_loss = 0.0
        running_transition_loss = 0.0
        running_boundary_type_loss = 0.0
        running_boundary_loss = 0.0
        running_positive_rate_reg = 0.0
        epoch_micro_steps = 0
        train_loader_iter = iter(train_loader)

        for step in progress:
            if curriculum_enabled:
                progress_ratio = optimizer_step / max(total_optimizer_steps_target, 1)
                batch = sample_curriculum_batch(
                    dataset=train_dataset,
                    collator=collator,
                    batch_size=args.batch_size,
                    progress_ratio=progress_ratio,
                    rng=curriculum_rng,
                )
            else:
                batch = next(train_loader_iter)
            batch = move_batch_to_device(batch, device)
            if curriculum_enabled and args.condition_dropout_probability > 0:
                batch = apply_condition_dropout_to_batch(
                    batch=batch,
                    dropout_probability=args.condition_dropout_probability,
                    pad_token_id=tokenizer.pad_token_id,
                    rng=dropout_rng,
                )
            noisy_inputs = sample_noisy_inputs(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                mask_id=mask_id,
                max_noise_ratio=args.max_noise_ratio,
            )

            outputs = llada_model(
                input_ids=noisy_inputs,
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1].to(torch.float32)
            confidence_weights = build_confidence_weights(batch["label_confidence"], device=device)
            scaled_phase_mask = batch["phase_loss_mask"] * confidence_weights
            scaled_boundary_mask = batch["boundary_loss_mask"] * confidence_weights
            transition_labels, transition_loss_mask = derive_transition_targets(
                batch["phase_labels"],
                batch["phase_loss_mask"],
                batch["boundary_loss_mask"],
            )
            progress_ratio = optimizer_step / max(total_optimizer_steps_target, 1)
            training_stage = resolve_training_stage(progress_ratio)
            teacher_forcing_ratio = resolve_phase_teacher_forcing_ratio(progress_ratio)
            if hasattr(boundary_head, "predict_phase_logits"):
                shared_states = boundary_head.encode(hidden_states)
                phase_logits = boundary_head.predict_phase_logits(shared_states=shared_states)
                predicted_phase_posteriors = boundary_head.predict_phase_posteriors(
                    shared_states=shared_states,
                    phase_logits=phase_logits,
                )
                transition_logits = boundary_head.predict_transition_logits(shared_states=shared_states)
                phase_loss = masked_phase_cross_entropy(
                    logits=phase_logits,
                    labels=batch["phase_labels"],
                    loss_mask=scaled_phase_mask,
                )
                transition_loss = masked_asymmetric_focal_loss(
                    logits=transition_logits,
                    labels=transition_labels,
                    loss_mask=transition_loss_mask * confidence_weights,
                    gamma_pos=1.0,
                    gamma_neg=2.0,
                )
                typed_transition_logits = (
                    boundary_head.predict_typed_transition_logits(shared_states=shared_states)
                    if hasattr(boundary_head, "predict_typed_transition_logits")
                    else None
                )
                boundary_type_logits = (
                    boundary_head.predict_boundary_type_logits(shared_states=shared_states)
                    if hasattr(boundary_head, "predict_boundary_type_logits")
                    else None
                )
                boundary_type_loss = torch.zeros((), device=device)
                boundary_type_posteriors = None
                if typed_transition_logits is not None:
                    boundary_type_loss = boundary_type_loss + masked_phase_cross_entropy(
                        logits=typed_transition_logits,
                        labels=batch["typed_transition_labels"],
                        loss_mask=batch["transition_loss_mask"] * confidence_weights,
                    )
                if boundary_type_logits is not None:
                    boundary_type_loss = boundary_type_loss + masked_phase_cross_entropy(
                        logits=boundary_type_logits,
                        labels=batch["boundary_type_labels"],
                        loss_mask=batch["boundary_type_loss_mask"] * confidence_weights,
                    )
                    predicted_boundary_type_posteriors = torch.softmax(boundary_type_logits, dim=-1)
                    gold_boundary_type_posteriors = F.one_hot(
                        batch["boundary_type_labels"].clamp(min=0),
                        num_classes=int(boundary_head.boundary_type_count),
                    ).to(torch.float32)
                    boundary_type_posteriors = (
                        teacher_forcing_ratio * gold_boundary_type_posteriors
                        + (1.0 - teacher_forcing_ratio) * predicted_boundary_type_posteriors
                    )
                gold_phase_posteriors = build_gold_phase_posteriors(
                    batch["phase_labels"],
                    batch["phase_loss_mask"],
                    num_labels=len(boundary_head.config.phase_label_vocab),
                ).to(device)
                gold_transition_condition = transition_labels
                if training_stage == "stage_a":
                    conditioned_phase = gold_phase_posteriors
                    conditioned_transition = gold_transition_condition
                    boundary_logits = torch.zeros_like(batch["boundary_labels"])
                    boundary_loss = torch.zeros((), device=device)
                    positive_rate_reg = torch.zeros((), device=device)
                    total_loss = (
                        getattr(boundary_head.config, "phase_loss_weight", 0.5) * phase_loss
                        + getattr(boundary_head.config, "transition_loss_weight", 0.75) * transition_loss
                        + getattr(boundary_head.config, "boundary_type_loss_weight", 0.0) * boundary_type_loss
                    )
                else:
                    conditioned_phase = (
                        teacher_forcing_ratio * gold_phase_posteriors
                        + (1.0 - teacher_forcing_ratio) * predicted_phase_posteriors
                    )
                    conditioned_transition = (
                        teacher_forcing_ratio * gold_transition_condition
                        + (1.0 - teacher_forcing_ratio) * torch.sigmoid(transition_logits)
                    )
                    boundary_logits = boundary_head.predict_joint_boundary_logits(
                        shared_states=shared_states,
                        phase_posteriors=conditioned_phase,
                        transition_condition=conditioned_transition,
                        boundary_type_posteriors=boundary_type_posteriors,
                    )
                    boundary_loss = masked_asymmetric_focal_loss(
                        logits=boundary_logits,
                        labels=batch["boundary_labels"],
                        loss_mask=scaled_boundary_mask,
                        pos_weight=boundary_pos_weight,
                    )
                    positive_rate_reg = boundary_positive_rate_regularizer(
                        logits=boundary_logits,
                        loss_mask=scaled_boundary_mask,
                        target_positive_rate=target_boundary_positive_rate,
                    )
                    total_loss = (
                        getattr(boundary_head.config, "phase_loss_weight", 0.5) * phase_loss
                        + getattr(boundary_head.config, "transition_loss_weight", 0.75) * transition_loss
                        + getattr(boundary_head.config, "boundary_type_loss_weight", 0.0) * boundary_type_loss
                        + getattr(boundary_head.config, "boundary_loss_weight", 1.0) * boundary_loss
                        + (float(args.positive_rate_reg_weight) * positive_rate_reg)
                    )
            else:
                boundary_logits = boundary_head(hidden_states)
                boundary_loss = masked_asymmetric_focal_loss(
                    logits=boundary_logits,
                    labels=batch["boundary_labels"],
                    loss_mask=scaled_boundary_mask,
                    pos_weight=boundary_pos_weight,
                )
                phase_loss = torch.zeros((), device=device)
                transition_loss = torch.zeros((), device=device)
                boundary_type_loss = torch.zeros((), device=device)
                positive_rate_reg = boundary_positive_rate_regularizer(
                    logits=boundary_logits,
                    loss_mask=scaled_boundary_mask,
                    target_positive_rate=target_boundary_positive_rate,
                )
                total_loss = boundary_loss + (float(args.positive_rate_reg_weight) * positive_rate_reg)

            loss = total_loss / args.gradient_accumulation_steps
            loss.backward()
            micro_step += 1
            epoch_micro_steps += 1

            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(boundary_head.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                if should_save_step_checkpoint(
                    optimizer_step=optimizer_step,
                    save_every_steps=resolved_save_every_steps,
                    checkpoint_steps=configured_checkpoint_steps,
                ):
                    save_current_step_checkpoint()
                if args.max_train_steps is not None and optimizer_step >= args.max_train_steps:
                    stop_training = True
            running_loss += float(total_loss.item())
            running_phase_loss += float(phase_loss.item())
            running_transition_loss += float(transition_loss.item())
            running_boundary_type_loss += float(boundary_type_loss.item())
            running_boundary_loss += float(boundary_loss.item())
            running_positive_rate_reg += float(positive_rate_reg.item())
            progress.set_postfix(
                loss=running_loss / max(epoch_micro_steps, 1),
                phase_loss=running_phase_loss / max(epoch_micro_steps, 1),
                transition_loss=running_transition_loss / max(epoch_micro_steps, 1),
                type_loss=running_boundary_type_loss / max(epoch_micro_steps, 1),
                boundary_loss=running_boundary_loss / max(epoch_micro_steps, 1),
                optimizer_step=optimizer_step,
            )
            if stop_training:
                break

        if not stop_training and epoch_micro_steps % args.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(boundary_head.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            if should_save_step_checkpoint(
                optimizer_step=optimizer_step,
                save_every_steps=resolved_save_every_steps,
                checkpoint_steps=configured_checkpoint_steps,
            ):
                save_current_step_checkpoint()
            if args.max_train_steps is not None and optimizer_step >= args.max_train_steps:
                stop_training = True

        metrics = {
            "epoch": epoch,
            "train_loss": running_loss / max(epoch_micro_steps, 1),
            "train_phase_loss": running_phase_loss / max(epoch_micro_steps, 1),
            "train_transition_loss": running_transition_loss / max(epoch_micro_steps, 1),
            "train_boundary_type_loss": running_boundary_type_loss / max(epoch_micro_steps, 1),
            "train_boundary_loss": running_boundary_loss / max(epoch_micro_steps, 1),
            "train_boundary_positive_rate_reg": running_positive_rate_reg / max(epoch_micro_steps, 1),
            "optimizer_step": optimizer_step,
            "micro_step": micro_step,
        }

        if valid_loader is not None:
            val_metrics = evaluate(
                llada_model=llada_model,
                boundary_head=boundary_head,
                dataloader=valid_loader,
                mask_id=mask_id,
                max_noise_ratio=args.max_noise_ratio,
                device=device,
                phase_label_vocab=train_dataset.phase_label_vocab,
                target_boundary_positive_rate=target_boundary_positive_rate,
            )
            metrics.update({f"valid_{key}": value for key, value in val_metrics.items()})

        history.append(metrics)

        save_current_step_checkpoint()
        metadata = build_training_metadata(
            args=args,
            history=history,
            include_terminal_boundary=include_terminal_boundary,
            optimizer_step=optimizer_step,
            micro_step=micro_step,
            selection_metric_source=selection_metric_source,
            save_every_steps=resolved_save_every_steps,
            checkpoint_steps=configured_checkpoint_steps,
            target_boundary_positive_rate=target_boundary_positive_rate,
            positive_rate_source=positive_rate_source,
        )

        save_boundary_head(
            boundary_head,
            output_dir / "boundary_head_last.pt",
            metadata=metadata,
        )
        if not step_selection_enabled:
            save_boundary_head(
                boundary_head,
                output_dir / "boundary_head_best.pt",
                metadata=metadata,
            )

        with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)

        if stop_training:
            break

    print("Training finished.")
    if step_selection_enabled:
        print("Checkpoint selection metric: proxy_eval_required")
        print("Use run_checkpoint_proxy_sweep.py to materialize boundary_head_best.pt from the best step checkpoint.")
    else:
        print("Checkpoint selection metric: training_last_checkpoint")
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
