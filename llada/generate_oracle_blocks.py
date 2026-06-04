from __future__ import annotations

import torch
import torch.nn.functional as F


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_transfer_index(
    logits: torch.Tensor,
    predicted_tokens: torch.Tensor,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    num_transfer_tokens,
    threshold: float | None = None,
):
    x0 = predicted_tokens

    if remasking == "low_confidence":
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)

    if threshold is not None:
        transfer_index = mask_index & (confidence >= threshold)
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
        transfer_index = (transfer_index | force_mask) & mask_index
        return x0, transfer_index

    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    _, idx = torch.sort(confidence, dim=1, descending=True)
    batch_size, seq_len = confidence.shape
    cols = torch.arange(seq_len, device=confidence.device).unsqueeze(0).expand(batch_size, seq_len)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(batch_size, seq_len)
    select_sorted = cols < k_expanded

    transfer_int = torch.zeros(batch_size, seq_len, device=confidence.device, dtype=torch.int8)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index
    return x0, transfer_index


def _oracle_block_length(
    oracle_block_sizes,
    block_idx: int,
    init_block_length: int,
    gen_length: int,
    generated_length: int,
) -> int:
    remaining_length = gen_length - generated_length
    if remaining_length <= 0:
        return 0
    if oracle_block_sizes and block_idx < len(oracle_block_sizes):
        return min(int(oracle_block_sizes[block_idx]), remaining_length)
    return min(int(init_block_length), remaining_length)


def _init_generation_tensor(model, prompt: torch.Tensor, gen_length: int, mask_id: int) -> torch.Tensor:
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    return x


def _normalize_boundary_carry_mask(oracle_block_sizes, boundary_carry_mask):
    block_sizes = [max(1, int(size)) for size in list(oracle_block_sizes or [])]
    expected_count = max(0, len(block_sizes) - 1)
    normalized_mask = [1 if bool(value) else 0 for value in list(boundary_carry_mask or [])]
    if len(normalized_mask) != expected_count:
        raise ValueError(
            f"boundary_carry_mask must have length {expected_count} for oracle_block_sizes of length {len(block_sizes)}"
        )
    return normalized_mask


@torch.no_grad()
def generate_oracle_blocks(
    model,
    prompt,
    steps=128,
    gen_length=128,
    init_block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=None,
    oracle_block_sizes=None,
    **_ignored_kwargs,
):
    assert prompt.shape[0] == 1, "Batch size > 1 is not supported"
    assert threshold is not None, "threshold must be set (e.g. threshold=0.9 or threshold=1.0 for top-1)"

    x = _init_generation_tensor(model, prompt, gen_length, mask_id)
    generated_length = 0
    nfe_history = []
    block_history = []
    block_idx = 0

    while generated_length < gen_length:
        nfe = 0
        output = model(x)
        logits = output.logits
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)
        nfe += 1

        block_length = _oracle_block_length(oracle_block_sizes, block_idx, init_block_length, gen_length, generated_length)
        block_history.append(block_length)
        block_idx += 1

        block_start = prompt.shape[1] + generated_length
        block_end = block_start + block_length
        generated_length += block_length

        mask_index = x == mask_id
        mask_index[:, block_end:] = 0
        x0, transfer_index = get_transfer_index(logits, predicted_tokens, remasking, mask_index, x, None, threshold)
        x[transfer_index] = x0[transfer_index]

        while True:
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break
            mask_index = x == mask_id
            mask_index[:, block_end:] = 0
            block_output = model(x)
            block_logits = block_output.logits
            block_logits_with_noise = add_gumbel_noise(block_logits, temperature=temperature)
            block_predicted_tokens = torch.argmax(block_logits_with_noise, dim=-1)
            nfe += 1
            x0, transfer_index = get_transfer_index(
                block_logits,
                block_predicted_tokens,
                remasking,
                mask_index,
                x,
                None,
                threshold,
            )
            x[transfer_index] = x0[transfer_index]
        nfe_history.append(nfe)

    return x, nfe_history, block_history


@torch.no_grad()
def generate_oracle_blocks_prefix_cache(
    model,
    prompt,
    steps=128,
    gen_length=128,
    init_block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=None,
    oracle_block_sizes=None,
    **_ignored_kwargs,
):
    assert prompt.shape[0] == 1, "Batch size > 1 is not supported"
    assert threshold is not None, "threshold must be set (e.g. threshold=0.9 or threshold=1.0 for top-1)"

    x = _init_generation_tensor(model, prompt, gen_length, mask_id)
    generated_length = 0
    nfe_history = []
    block_history = []
    block_idx = 0

    while generated_length < gen_length:
        nfe = 0
        output = model(x, use_cache=True)
        full_cache = output.past_key_values
        logits = output.logits
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)
        nfe += 1

        block_length = _oracle_block_length(oracle_block_sizes, block_idx, init_block_length, gen_length, generated_length)
        block_history.append(block_length)
        block_idx += 1

        block_start = prompt.shape[1] + generated_length
        block_end = block_start + block_length
        generated_length += block_length

        mask_index = x == mask_id
        mask_index[:, block_end:] = 0
        x0, transfer_index = get_transfer_index(logits, predicted_tokens, remasking, mask_index, x, None, threshold)
        x[transfer_index] = x0[transfer_index]

        prefix_cache = []
        for layer_cache in full_cache:
            truncated = ()
            for tensor in layer_cache:
                truncated += (tensor[:, :, :block_start],)
            prefix_cache.append(truncated)

        while True:
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break
            mask_index = x[:, block_start:] == mask_id
            mask_index[:, block_length:] = 0
            block_output = model(x[:, block_start:], past_key_values=prefix_cache, use_cache=True)
            block_logits = block_output.logits
            block_logits_with_noise = add_gumbel_noise(block_logits, temperature=temperature)
            block_predicted_tokens = torch.argmax(block_logits_with_noise, dim=-1)
            nfe += 1
            x0, transfer_index = get_transfer_index(
                block_logits,
                block_predicted_tokens,
                remasking,
                mask_index,
                x[:, block_start:],
                None,
                threshold,
            )
            x[:, block_start:][transfer_index] = x0[transfer_index]
        nfe_history.append(nfe)

    return x, nfe_history, block_history


@torch.no_grad()
def generate_oracle_blocks_dual_cache(
    model,
    prompt,
    steps=128,
    gen_length=128,
    init_block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=None,
    oracle_block_sizes=None,
    **_ignored_kwargs,
):
    assert prompt.shape[0] == 1, "Batch size > 1 is not supported"
    assert threshold is not None, "threshold must be set (e.g. threshold=0.9 or threshold=1.0 for top-1)"

    x = _init_generation_tensor(model, prompt, gen_length, mask_id)
    generated_length = 0
    nfe_history = []
    block_history = []
    block_idx = 0

    while generated_length < gen_length:
        nfe = 0
        output = model(x, use_cache=True)
        full_cache = output.past_key_values
        logits = output.logits
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)
        nfe += 1

        block_length = _oracle_block_length(oracle_block_sizes, block_idx, init_block_length, gen_length, generated_length)
        block_history.append(block_length)
        block_idx += 1

        block_start = prompt.shape[1] + generated_length
        block_end = block_start + block_length
        generated_length += block_length

        mask_index = x == mask_id
        mask_index[:, block_end:] = 0
        x0, transfer_index = get_transfer_index(logits, predicted_tokens, remasking, mask_index, x, None, threshold)
        x[transfer_index] = x0[transfer_index]

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, block_start:block_end] = 1
        while True:
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break
            mask_index = x[:, block_start:block_end] == mask_id
            block_output = model(
                x[:, block_start:block_end],
                past_key_values=full_cache,
                use_cache=True,
                replace_position=replace_position,
            )
            block_logits = block_output.logits
            block_logits_with_noise = add_gumbel_noise(block_logits, temperature=temperature)
            block_predicted_tokens = torch.argmax(block_logits_with_noise, dim=-1)
            nfe += 1
            x0, transfer_index = get_transfer_index(
                block_logits,
                block_predicted_tokens,
                remasking,
                mask_index,
                x[:, block_start:block_end],
                None,
                threshold,
            )
            x[:, block_start:block_end][transfer_index] = x0[transfer_index]
        nfe_history.append(nfe)

    return x, nfe_history, block_history


@torch.no_grad()
def generate_oracle_blocks_boundary_gate(
    model,
    prompt,
    steps=128,
    gen_length=128,
    init_block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=None,
    oracle_block_sizes=None,
    boundary_carry_mask=None,
    **_ignored_kwargs,
):
    assert prompt.shape[0] == 1, "Batch size > 1 is not supported"
    assert threshold is not None, "threshold must be set (e.g. threshold=0.9 or threshold=1.0 for top-1)"

    carry_mask = _normalize_boundary_carry_mask(oracle_block_sizes, boundary_carry_mask)
    x = _init_generation_tensor(model, prompt, gen_length, mask_id)
    generated_length = 0
    nfe_history = []
    block_history = []
    block_idx = 0

    while generated_length < gen_length:
        nfe = 0
        output = model(x, use_cache=True)
        full_cache = output.past_key_values
        logits = output.logits
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)
        nfe += 1

        current_block_idx = block_idx
        block_length = _oracle_block_length(
            oracle_block_sizes,
            current_block_idx,
            init_block_length,
            gen_length,
            generated_length,
        )
        block_history.append(block_length)
        block_idx += 1

        block_start = prompt.shape[1] + generated_length
        block_end = block_start + block_length
        generated_length += block_length

        mask_index = x == mask_id
        mask_index[:, block_end:] = 0
        x0, transfer_index = get_transfer_index(logits, predicted_tokens, remasking, mask_index, x, None, threshold)
        x[transfer_index] = x0[transfer_index]

        use_prefix_cache = current_block_idx > 0 and bool(carry_mask[current_block_idx - 1])
        if use_prefix_cache:
            prefix_cache = []
            for layer_cache in full_cache:
                truncated = ()
                for tensor in layer_cache:
                    truncated += (tensor[:, :, :block_start],)
                prefix_cache.append(truncated)

        while True:
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break
            if use_prefix_cache:
                mask_index = x[:, block_start:] == mask_id
                mask_index[:, block_length:] = 0
                block_output = model(x[:, block_start:], past_key_values=prefix_cache, use_cache=True)
                block_logits = block_output.logits
                block_logits_with_noise = add_gumbel_noise(block_logits, temperature=temperature)
                block_predicted_tokens = torch.argmax(block_logits_with_noise, dim=-1)
                x0, transfer_index = get_transfer_index(
                    block_logits,
                    block_predicted_tokens,
                    remasking,
                    mask_index,
                    x[:, block_start:],
                    None,
                    threshold,
                )
                x[:, block_start:][transfer_index] = x0[transfer_index]
            else:
                mask_index = x == mask_id
                mask_index[:, block_end:] = 0
                block_output = model(x, use_cache=True)
                block_logits = block_output.logits
                block_logits_with_noise = add_gumbel_noise(block_logits, temperature=temperature)
                block_predicted_tokens = torch.argmax(block_logits_with_noise, dim=-1)
                x0, transfer_index = get_transfer_index(
                    block_logits,
                    block_predicted_tokens,
                    remasking,
                    mask_index,
                    x,
                    None,
                    threshold,
                )
                x[transfer_index] = x0[transfer_index]
            nfe += 1
        nfe_history.append(nfe)

    return x, nfe_history, block_history


__all__ = [
    "generate_oracle_blocks",
    "generate_oracle_blocks_prefix_cache",
    "generate_oracle_blocks_dual_cache",
    "generate_oracle_blocks_boundary_gate",
]
