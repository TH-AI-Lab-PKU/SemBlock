# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
# Copyright 2025 Guanxi Lu, Imperial College London (modifications)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA
# Modified by Guanxi Lu, Imperial College London

import random
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM
from semantic_boundary import score_boundary_window
from prior_boundary import choose_boundary_length_from_scores, compute_block_length_prior
from oracle_boundary import extract_oracle_block_sizes

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def compute_block_length(
    logits,                
    predicted_tokens,      
    prompt,                
    gen_length,       
    generated_length,
    default_block_length,
    delimiter_ids=[198],  # default: newline token (11=comma, 13=period, 198=newline)
    delimiter_threshold=float('inf'),
    boundary_head=None,
    hidden_states=None,
    boundary_prior_weight=0.7,
    boundary_prior_threshold=None,
    boundary_window_ratio=0.25,
):
    """
    Compute adaptive block length based on delimiter confidence and an optional
    semantic boundary prior.

    When boundary_head is provided, the scheduler first scores every token in
    the look-ahead window with the learned semantic boundary prior and blends
    it with delimiter confidence. Otherwise it falls back to the original
    delimiter-only AdaBlock behavior.
    """
    prompt_length = prompt.shape[1]
    block_start = prompt_length + generated_length
    remaining_length = gen_length - generated_length
    if delimiter_threshold is None:
        delimiter_threshold = float('inf')

    # Create sampling window (ratio of gen_length, capped by remaining)
    window_size = min(max(int(boundary_window_ratio * gen_length), 1), remaining_length)
    window_tokens = predicted_tokens[0, block_start:block_start + window_size]

    default_block_length = min(default_block_length, remaining_length)

    # Create mask for delimiter tokens
    delimiter_mask = torch.zeros_like(window_tokens, dtype=torch.bool)
    for token_id in delimiter_ids:
        delimiter_mask |= (window_tokens == token_id)

    delimiter_scores = torch.zeros(window_size, device=logits.device, dtype=torch.float32)
    if torch.any(delimiter_mask):
        window_positions = torch.arange(block_start, block_start + window_size, device=logits.device)
        window_logits = logits[0, window_positions, :]
        chosen_token_ids = predicted_tokens[0, window_positions].unsqueeze(-1)
        chosen_logits = torch.gather(window_logits, dim=-1, index=chosen_token_ids).squeeze(-1)
        chosen_confidence = torch.exp(chosen_logits - torch.logsumexp(window_logits, dim=-1))
        delimiter_scores = torch.where(delimiter_mask, chosen_confidence.to(torch.float32), delimiter_scores)

    if boundary_head is not None:
        if hidden_states is None:
            raise ValueError("hidden_states are required when boundary_head is provided.")
        boundary_scores = score_boundary_window(
            boundary_head=boundary_head,
            hidden_states=hidden_states,
            start=block_start,
            end=block_start + window_size,
        )[0].to(device=logits.device, dtype=torch.float32)
        mix = float(boundary_prior_weight)
        mix = max(0.0, min(1.0, mix))
        combined_scores = mix * boundary_scores + (1.0 - mix) * delimiter_scores
        best_score, best_idx = torch.max(combined_scores, dim=0)
        threshold = boundary_prior_threshold
        if threshold is None:
            threshold = delimiter_threshold
        if best_score.item() >= threshold:
            return int(best_idx.item()) + 1

    # Original delimiter-only fallback.
    if not torch.any(delimiter_mask):
        return default_block_length

    max_confidence, best_idx = torch.max(delimiter_scores, dim=0)
    if max_confidence.item() >= delimiter_threshold:
        return int(best_idx.item()) + 1
    return default_block_length


def compute_block_length_semantic_prior(
    prompt,
    gen_length,
    generated_length,
    default_block_length,
    boundary_head,
    hidden_states,
    boundary_prior_threshold=None,
    boundary_window_ratio=0.25,
):
    if hidden_states is None:
        raise ValueError("hidden_states are required when semantic prior scheduling is enabled.")

    prompt_length = prompt.shape[1]
    block_start = prompt_length + generated_length
    remaining_length = gen_length - generated_length
    cap = min(default_block_length, remaining_length)
    if cap <= 0:
        return cap

    window_size = min(max(int(boundary_window_ratio * gen_length), 1), remaining_length, cap)
    boundary_scores = score_boundary_window(
        boundary_head=boundary_head,
        hidden_states=hidden_states,
        start=block_start,
        end=block_start + window_size,
    )[0].to(dtype=torch.float32).detach().cpu().tolist()
    threshold = 0.5 if boundary_prior_threshold is None else boundary_prior_threshold
    return choose_boundary_length_from_scores(
        boundary_scores,
        default_block_length=cap,
        threshold=threshold,
    )


def maybe_degrade_block_length(
    block_length,
    *,
    mode='none',
    strength=0,
    block_idx=0,
    remaining_length=0,
    default_block_length=0,
):
    mode = str(mode or 'none').lower()
    strength = int(float(strength or 0))
    if mode in ('none', 'off', '') or strength <= 0:
        return int(block_length)

    cap = max(1, int(remaining_length or default_block_length or block_length))
    default_cap = max(1, min(int(default_block_length or cap), cap))
    block_length = max(1, min(int(block_length), cap))

    if mode == 'early':
        degraded = block_length - strength
    elif mode == 'late':
        degraded = block_length + strength
    elif mode == 'jitter':
        degraded = block_length + (strength if block_idx % 2 == 0 else -strength)
    elif mode == 'oversegment':
        degraded = max(1, block_length // max(2, strength))
    elif mode == 'random':
        degraded = random.randint(1, default_cap)
    else:
        degraded = block_length
    return max(1, min(int(degraded), cap))


@ torch.no_grad()
def generate_adablock(model, prompt, steps=128, gen_length=128, init_block_length=128, temperature=0.,
            remasking='low_confidence', mask_id=126336, threshold=None, delimiter_ids=[198], delimiter_threshold=float('inf'),
            boundary_head=None, boundary_prior_weight=0.7, boundary_prior_threshold=None, boundary_window_ratio=0.25,
            block_strategy='adablock', task_type='nl', tokenizer=None, oracle_block_sizes=None,
            prior_fallback_prob=0.0, boundary_degrade_mode='none', boundary_degrade_strength=0):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        init_block_length: The block length for the first block; for subsequent blocks, the block length is computed adaptively.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        threshold: Threshold for top-k sampling.
        delimiter_ids: List of token ids used as delimiters for adaptive block length.
        delimiter_threshold: Confidence threshold for block length prediction.
        boundary_head: Optional semantic boundary predictor trained on top of LLaDA hidden states.
        boundary_prior_weight: Blend weight for semantic prior vs. delimiter confidence.
        boundary_prior_threshold: Acceptance threshold for the blended semantic prior score.
        boundary_window_ratio: Look-ahead window ratio used by the boundary scheduler.
        block_strategy: 'fixed' | 'adablock' | 'prior' | 'oracle'.  Selects the block-size scheduler.
        task_type: 'code' | 'nl' | 'math'.  Used only when block_strategy='prior'.
        tokenizer: HuggingFace tokenizer.  Required when block_strategy='prior'.
        oracle_block_sizes: Pre-computed list of block sizes (ints).  Required when
            block_strategy='oracle'.  Derived from gold reference answers.  Falls back
            to init_block_length when the list is exhausted.
    '''
    assert prompt.shape[0] == 1, "Batch size > 1 is not supported"
    
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    # Block size: fixed (delimiter_threshold=inf) or adaptive (delimiter_threshold<inf, e.g., 0.3)
    # Token transfer: threshold-based (threshold<1) or top-1 (threshold=1.0); top-k (k>1) is not supported
    assert threshold is not None, "threshold must be set (e.g., threshold=0.9 or threshold=1.0 for top-1)"

    generated_length = 0
    nfe_history = []
    block_history = []
    block_idx = 0
    while generated_length < gen_length:
        nfe = 0

        output = model(x, output_hidden_states=boundary_head is not None)
        logits = output.logits
        hidden_states = output.hidden_states[-1] if boundary_head is not None and output.hidden_states is not None else None
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)
        nfe += 1

        if block_strategy == 'oracle':
            remaining_length = gen_length - generated_length
            if oracle_block_sizes and block_idx < len(oracle_block_sizes):
                block_length = min(oracle_block_sizes[block_idx], remaining_length)
            else:
                block_length = min(init_block_length, remaining_length)
        elif block_strategy == 'prior':
            assert tokenizer is not None, "tokenizer must be provided when block_strategy='prior'"
            if prior_fallback_prob > 0.0 and random.random() < prior_fallback_prob:
                block_length = compute_block_length(
                    logits,
                    predicted_tokens,
                    prompt,
                    gen_length,
                    generated_length,
                    init_block_length,
                    delimiter_ids=delimiter_ids,
                    delimiter_threshold=delimiter_threshold,
                    boundary_head=boundary_head,
                    hidden_states=hidden_states,
                    boundary_prior_weight=boundary_prior_weight,
                    boundary_prior_threshold=boundary_prior_threshold,
                    boundary_window_ratio=boundary_window_ratio,
                )
            else:
                block_length = compute_block_length_prior(
                    predicted_tokens,
                    prompt,
                    gen_length,
                    generated_length,
                    init_block_length,
                    tokenizer,
                    task_type=task_type,
                    boundary_window_ratio=boundary_window_ratio,
                )
        else:
            # 'fixed' forces delimiter_threshold=inf (no adaptive sizing);
            # 'adablock' uses the caller-supplied delimiter_threshold value.
            _eff_threshold = float('inf') if block_strategy == 'fixed' else delimiter_threshold
            block_length = compute_block_length(
                logits,
                predicted_tokens,
                prompt,
                gen_length,
                generated_length,
                init_block_length,
                delimiter_ids=delimiter_ids,
                delimiter_threshold=_eff_threshold,
                boundary_head=boundary_head,
                hidden_states=hidden_states,
                boundary_prior_weight=boundary_prior_weight,
                boundary_prior_threshold=boundary_prior_threshold,
                boundary_window_ratio=boundary_window_ratio,
            )

        block_length = maybe_degrade_block_length(
            block_length,
            mode=boundary_degrade_mode,
            strength=boundary_degrade_strength,
            block_idx=block_idx,
            remaining_length=gen_length - generated_length,
            default_block_length=init_block_length,
        )

        block_history.append(block_length)
        block_idx += 1

        block_start = prompt.shape[1] + generated_length
        block_end = block_start + block_length
        generated_length += block_length

        # only allow transfer tokens in current block
        mask_index = (x == mask_id)
        mask_index[:, block_end:] = 0

        x0, transfer_index = get_transfer_index(logits, predicted_tokens, remasking, mask_index, x, None, threshold)
        x[transfer_index] = x0[transfer_index]

        while True:
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break
            mask_index = (x == mask_id)
            mask_index[:, block_end:] = 0
            block_output = model(x)
            block_logits = block_output.logits
            block_logits_with_noise = add_gumbel_noise(block_logits, temperature=temperature)
            block_predicted_tokens = torch.argmax(block_logits_with_noise, dim=-1)
            nfe += 1
            x0, transfer_index = get_transfer_index(block_logits, block_predicted_tokens, remasking, mask_index,
                                            x, None, threshold)
            x[transfer_index] = x0[transfer_index]
        nfe_history.append(nfe)

    return x, nfe_history, block_history

@torch.no_grad()
def generate_adablock_prefix_cache(model, prompt, steps=128, gen_length=128, init_block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, delimiter_ids=[198], delimiter_threshold=float('inf'),
             boundary_head=None, boundary_prior_weight=0.7, boundary_prior_threshold=None, boundary_window_ratio=0.25,
             block_strategy='adablock', task_type='nl', tokenizer=None, oracle_block_sizes=None,
             prior_fallback_prob=0.0, boundary_degrade_mode='none', boundary_degrade_strength=0):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        init_block_length: The block length for the first block; for subsequent blocks, the block length is computed adaptively.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        threshold: Threshold for top-k sampling.
        delimiter_ids: List of token ids used as delimiters for adaptive block length.
        delimiter_threshold: Confidence threshold for block length prediction.
        boundary_head: Optional semantic boundary predictor trained on top of LLaDA hidden states.
        boundary_prior_weight: Blend weight for semantic prior vs. delimiter confidence.
        boundary_prior_threshold: Acceptance threshold for the blended semantic prior score.
        boundary_window_ratio: Look-ahead window ratio used by the boundary scheduler.
        oracle_block_sizes: Pre-computed list of block sizes (ints). Required when block_strategy='oracle'.
    '''
    assert prompt.shape[0] == 1, "Batch size > 1 is not supported"

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    # Block size: fixed (delimiter_threshold=inf) or adaptive (delimiter_threshold<inf, e.g., 0.3)
    # Token transfer: threshold-based (threshold<1) or top-1 (threshold=1.0); top-k (k>1) is not supported
    assert threshold is not None, "threshold must be set (e.g., threshold=0.9 or threshold=1.0 for top-1)"

    generated_length = 0
    nfe_history = []
    block_history = []
    block_idx = 0

    while generated_length < gen_length:
        nfe = 0

        output = model(x, use_cache=True, output_hidden_states=boundary_head is not None)
        full_cache = output.past_key_values
        logits = output.logits
        hidden_states = output.hidden_states[-1] if boundary_head is not None and output.hidden_states is not None else None
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)
        nfe += 1

        if block_strategy == 'oracle':
            remaining_length = gen_length - generated_length
            if oracle_block_sizes and block_idx < len(oracle_block_sizes):
                block_length = min(oracle_block_sizes[block_idx], remaining_length)
            else:
                block_length = min(init_block_length, remaining_length)
        elif block_strategy == 'prior':
            assert tokenizer is not None, "tokenizer must be provided when block_strategy='prior'"
            if prior_fallback_prob > 0.0 and random.random() < prior_fallback_prob:
                block_length = compute_block_length(
                    logits,
                    predicted_tokens,
                    prompt,
                    gen_length,
                    generated_length,
                    init_block_length,
                    delimiter_ids=delimiter_ids,
                    delimiter_threshold=delimiter_threshold,
                    boundary_head=boundary_head,
                    hidden_states=hidden_states,
                    boundary_prior_weight=boundary_prior_weight,
                    boundary_prior_threshold=boundary_prior_threshold,
                    boundary_window_ratio=boundary_window_ratio,
                )
            else:
                block_length = compute_block_length_prior(
                    predicted_tokens,
                    prompt,
                    gen_length,
                    generated_length,
                    init_block_length,
                    tokenizer,
                    task_type=task_type,
                    boundary_window_ratio=boundary_window_ratio,
                )
        else:
            _eff_threshold = float('inf') if block_strategy == 'fixed' else delimiter_threshold
            block_length = compute_block_length(
                logits,
                predicted_tokens,
                prompt,
                gen_length,
                generated_length,
                init_block_length,
                delimiter_ids=delimiter_ids,
                delimiter_threshold=_eff_threshold,
                boundary_head=boundary_head,
                hidden_states=hidden_states,
                boundary_prior_weight=boundary_prior_weight,
                boundary_prior_threshold=boundary_prior_threshold,
                boundary_window_ratio=boundary_window_ratio,
            )

        block_length = maybe_degrade_block_length(
            block_length,
            mode=boundary_degrade_mode,
            strength=boundary_degrade_strength,
            block_idx=block_idx,
            remaining_length=gen_length - generated_length,
            default_block_length=init_block_length,
        )

        block_history.append(block_length)
        block_idx += 1

        block_start = prompt.shape[1] + generated_length
        block_end = block_start + block_length
        generated_length += block_length

        # only allow transfer tokens in current block
        mask_index = (x == mask_id)
        mask_index[:, block_end:] = 0

        x0, transfer_index = get_transfer_index(logits, predicted_tokens, remasking, mask_index, x, None, threshold)
        x[transfer_index] = x0[transfer_index]

        # truncate cache to prefix only (before current block)
        prefix_cache = []
        for i in range(len(full_cache)):
            prefix_cache.append(())
            for j in range(len(full_cache[i])):
                prefix_cache[i] += (full_cache[i][j][:, :, :block_start],)

        # 2nd and later denoising steps in block
        while True:
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break
            mask_index = (x[:, block_start:] == mask_id)
            mask_index[:, block_length:] = 0
            block_output = model(x[:, block_start:], past_key_values=prefix_cache, use_cache=True)
            block_logits = block_output.logits
            block_logits_with_noise = add_gumbel_noise(block_logits, temperature=temperature)
            block_predicted_tokens = torch.argmax(block_logits_with_noise, dim=-1)
            nfe += 1
            x0, transfer_index = get_transfer_index(block_logits, block_predicted_tokens, remasking, mask_index, 
                                            x[:, block_start:], None, threshold)
            x[:, block_start:][transfer_index] = x0[transfer_index]
        nfe_history.append(nfe)

    return x, nfe_history, block_history


@torch.no_grad()
def generate_adablock_dual_cache(model, prompt, steps=128, gen_length=128, init_block_length=128, temperature=0.,
            remasking='low_confidence', mask_id=126336, threshold=None, delimiter_ids=[198], delimiter_threshold=float('inf'),
            boundary_head=None, boundary_prior_weight=0.7, boundary_prior_threshold=None, boundary_window_ratio=0.25,
            block_strategy='adablock', task_type='nl', tokenizer=None, oracle_block_sizes=None,
            prior_fallback_prob=0.0, boundary_degrade_mode='none', boundary_degrade_strength=0):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        init_block_length: The block length for the first block; for subsequent blocks, the block length is computed adaptively.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        threshold: Threshold for top-k sampling.
        delimiter_ids: List of token ids used as delimiters for adaptive block length.
        delimiter_threshold: Confidence threshold for block length prediction.
        boundary_head: Optional semantic boundary predictor trained on top of LLaDA hidden states.
        boundary_prior_weight: Blend weight for semantic prior vs. delimiter confidence.
        boundary_prior_threshold: Acceptance threshold for the blended semantic prior score.
        boundary_window_ratio: Look-ahead window ratio used by the boundary scheduler.
        oracle_block_sizes: Pre-computed list of block sizes (ints). Required when block_strategy='oracle'.
    '''
    assert prompt.shape[0] == 1, "Batch size > 1 is not supported"

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    # Block size: fixed (delimiter_threshold=inf) or adaptive (delimiter_threshold<inf, e.g., 0.3)
    # Token transfer: threshold-based (threshold<1) or top-1 (threshold=1.0); top-k (k>1) is not supported
    assert threshold is not None, "threshold must be set (e.g., threshold=0.9 or threshold=1.0 for top-1)"

    generated_length = 0
    nfe_history = []
    block_history = []
    block_idx = 0

    while generated_length < gen_length:
        nfe = 0

        output = model(x, use_cache=True, output_hidden_states=boundary_head is not None)
        full_cache = output.past_key_values
        logits = output.logits
        hidden_states = output.hidden_states[-1] if boundary_head is not None and output.hidden_states is not None else None
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)
        nfe += 1

        if block_strategy == 'oracle':
            remaining_length = gen_length - generated_length
            if oracle_block_sizes and block_idx < len(oracle_block_sizes):
                block_length = min(oracle_block_sizes[block_idx], remaining_length)
            else:
                block_length = min(init_block_length, remaining_length)
        elif block_strategy == 'prior':
            assert tokenizer is not None, "tokenizer must be provided when block_strategy='prior'"
            if prior_fallback_prob > 0.0 and random.random() < prior_fallback_prob:
                block_length = compute_block_length(
                    logits,
                    predicted_tokens,
                    prompt,
                    gen_length,
                    generated_length,
                    init_block_length,
                    delimiter_ids=delimiter_ids,
                    delimiter_threshold=delimiter_threshold,
                    boundary_head=boundary_head,
                    hidden_states=hidden_states,
                    boundary_prior_weight=boundary_prior_weight,
                    boundary_prior_threshold=boundary_prior_threshold,
                    boundary_window_ratio=boundary_window_ratio,
                )
            else:
                block_length = compute_block_length_prior(
                    predicted_tokens,
                    prompt,
                    gen_length,
                    generated_length,
                    init_block_length,
                    tokenizer,
                    task_type=task_type,
                    boundary_window_ratio=boundary_window_ratio,
                )
        else:
            _eff_threshold = float('inf') if block_strategy == 'fixed' else delimiter_threshold
            block_length = compute_block_length(
                logits,
                predicted_tokens,
                prompt,
                gen_length,
                generated_length,
                init_block_length,
                delimiter_ids=delimiter_ids,
                delimiter_threshold=_eff_threshold,
                boundary_head=boundary_head,
                hidden_states=hidden_states,
                boundary_prior_weight=boundary_prior_weight,
                boundary_prior_threshold=boundary_prior_threshold,
                boundary_window_ratio=boundary_window_ratio,
            )

        block_length = maybe_degrade_block_length(
            block_length,
            mode=boundary_degrade_mode,
            strength=boundary_degrade_strength,
            block_idx=block_idx,
            remaining_length=gen_length - generated_length,
            default_block_length=init_block_length,
        )

        block_history.append(block_length)
        block_idx += 1

        block_start = prompt.shape[1] + generated_length
        block_end = block_start + block_length
        generated_length += block_length

        # only allow transfer tokens in current block
        mask_index = (x == mask_id)
        mask_index[:, block_end:] = 0

        x0, transfer_index = get_transfer_index(logits, predicted_tokens, remasking, mask_index, x, None, threshold)
        x[transfer_index] = x0[transfer_index]

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, block_start:block_end] = 1
        # 2nd and later denoising steps in block
        while True:
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break
            mask_index = (x[:, block_start:block_end] == mask_id)
            block_output = model(x[:, block_start:block_end], past_key_values=full_cache, use_cache=True, replace_position=replace_position)
            block_logits = block_output.logits
            block_logits_with_noise = add_gumbel_noise(block_logits, temperature=temperature)
            block_predicted_tokens = torch.argmax(block_logits_with_noise, dim=-1)
            nfe += 1
            x0, transfer_index = get_transfer_index(block_logits, block_predicted_tokens, remasking, mask_index, 
                                            x[:, block_start:block_end], None, threshold)
            x[:, block_start:block_end][transfer_index] = x0[transfer_index]
        nfe_history.append(nfe)

    return x, nfe_history, block_history

def get_transfer_index(
    logits: torch.Tensor,
    predicted_tokens: torch.Tensor,
    remasking: str,
    mask_index: torch.Tensor,   # (B, L) bool
    x: torch.Tensor,            # (B, L) long
    num_transfer_tokens,        # (B,) or (B,1) long tensor, or None when threshold is used
    threshold: float = None,
):
    """
    Returns:
        x0: (B, L) long — proposed tokens
        transfer_index: (B, L) bool — which positions to update this step
    """
    x0 = predicted_tokens  # (B, L)

    # Confidence for chosen tokens (or random)
    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, L), float64
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)  # (B, L)
    else:
        raise NotImplementedError(remasking)

    # Only modify masked spots; keep others as original x and set their confidence to -inf
    x0 = torch.where(mask_index, x0, x)

    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)  # (B, L)

    # Pick positions to transfer (vectorized)
    if threshold is not None:
        # Transfer all masked positions whose confidence >= threshold
        transfer_index = mask_index & (confidence >= threshold)

        # at least one token is transferred "always unmask max c^i"
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)  # (B, 1)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)

        # (Above Threshold) OR (Is Max Confidence)
        transfer_index = transfer_index | force_mask

        # Safety: do not unmask something that was not masked
        transfer_index = transfer_index & mask_index

        return x0, transfer_index

    # Else: per-row top-k with varying k (num_transfer_tokens), fully batched
    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

    # Ensure shape (B,) long
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    # Sort confidences descending (masked positions are valid; others are -inf)
    values, idx = torch.sort(confidence, dim=1, descending=True)

    B, L = confidence.shape
    # Build a mask that is True for the first k[b] columns in each row (sorted order)
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)   # (B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)                   # (B, L)
    select_sorted = cols < k_expanded                                            # (B, L) bool

    # Scatter the sorted True/False back to original column order
    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)  # (B, L)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index  # ensure we never select unmasked

    return x0, transfer_index

def main():
    device = 'cuda'

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out_ids, nfe_history, block_history = generate_adablock(model, input_ids, steps=32, gen_length=256, init_block_length=16, temperature=0., remasking='low_confidence', threshold=0.9, delimiter_ids=[198], delimiter_threshold=0.3)
    
    print(tokenizer.batch_decode(out_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0])
    print()
    print(f"NFE for each block: {nfe_history}")
    print(f"Total NFE: {sum(nfe_history)}")
    print(f"Average NFE per block: {sum(nfe_history) / len(nfe_history)}")
    print()
    print(f"Block length for each block: {block_history}")
    print(f"Number of blocks: {len(block_history)}")
    print(f"Average block length: {sum(block_history) / len(block_history)}")

if __name__ == '__main__':
    main()
