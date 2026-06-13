# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F


def _beam_step_select(mode,
                      logp_matrix,      # [B, act, n_digit*VOC]
                      cur_beam_logp,    # [B, act]
                      beam_ids,         # [B, act, n_digit]  (Parent nodes)
                      n_digit, VOC, beam_act,
                      rand_cfg):
    """
    Unified single-step branch selection logic.

    Args:
        mode: "confidence" or "random"
        logp_matrix: Log probability matrix of the current step [B, act, n_digit*VOC]
        cur_beam_logp: Log probability of the current beam [B, act]
        beam_ids: Token sequence of the current beam [B, act, n_digit]
        n_digit: Number of digits
        VOC: Vocabulary size
        beam_act: Number of active beams
        rand_cfg: Random sampling configuration dictionary

    Returns:
        next_lp: Log probability of the next step [B, act]
        next_ids: Token sequence of the next step [B, act, n_digit]
    """
    B = logp_matrix.size(0)

    if mode == "confidence":
        # Confidence mode: select the paths with the highest probabilities
        cand_lp  = cur_beam_logp.unsqueeze(-1) + logp_matrix      # logP
        flat_lp  = cand_lp.view(B, -1)
        best_lp, flat_idx = torch.topk(flat_lp, k=beam_act)       # [B, act]
    else:   # "random"
        # Random mode: use temperature and top-p/top-k sampling
        temperature = rand_cfg.get("temperature", 1.0)
        logits = (cur_beam_logp.unsqueeze(-1) + logp_matrix) / temperature      # [B, act, *]

        # top-k truncation
        top_k = rand_cfg.get("top_k")
        if top_k is not None:
            kth_vals, _ = logits.topk(top_k, dim=-1)
            min_valid   = kth_vals[..., -1:].detach()
            logits      = torch.where(logits < min_valid, logits.new_full((), -1e9), logits)

        # top-p (nucleus) sampling
        top_p = rand_cfg.get("top_p")
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens that exceed the threshold and subsequent tokens (keep at least one)
            sorted_indices_to_remove = cumsum_probs > top_p
            # Force keep the first position (avoid removing everything)
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # Restore the boolean mask to the original order
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        probs = torch.softmax(logits, dim=-1)                   # True probabilities
        flat_prob = probs.view(B, -1)

        # 🚀 Fix: Save the current random seed state to avoid polluting training
        original_state = torch.get_rng_state()
        try:
            # Fix seed (optional)
            seed = rand_cfg.get("seed")
            if seed is not None:
                torch.manual_seed(seed)

            flat_idx = torch.multinomial(flat_prob, beam_act, replacement=False)  # [B, act]
            idx_rows = torch.arange(B, device=flat_idx.device).unsqueeze(1)
            best_lp  = logits.view(B, -1)[idx_rows, flat_idx]        # Corresponding logP
        finally:
            # 🚀 Restore the original random seed state
            torch.set_rng_state(original_state)
    # -------------------------------------------------------------------------

    parent   = flat_idx // (n_digit * VOC)
    remain   = flat_idx %  (n_digit * VOC)
    d_pos    = remain // VOC
    tok      = remain %  VOC

    batch_idx = torch.arange(B, device=beam_ids.device).unsqueeze(1)
    next_ids  = beam_ids[batch_idx, parent].clone()
    next_ids.scatter_(2, d_pos.unsqueeze(-1), tok.unsqueeze(-1))
    return best_lp, next_ids


def expand_cross_kv_for_beams(initial_kv_cache, beam_size):
    """
    Copy the cross-KV obtained in the first step to each beam, while self-KV is still set to None;
    In this way, the self-attention KV of DecoderBlock will continue to accumulate, while cross-KV will not be recomputed.

    Args:
        initial_kv_cache: Initial KV cache
        beam_size: Beam size

    Returns:
        Expanded KV cache
    """
    if initial_kv_cache is None:
        return None

    expanded = []
    for layer_cache in initial_kv_cache:
        if layer_cache is None:
            expanded.append(None)
            continue

        self_kv, cross_kv = layer_cache        # self_kv only exists in the first step, subsequently accumulated by cache
        if cross_kv is not None:
            k, v = cross_kv                    # [B, S, d]
            k = k.unsqueeze(1).repeat(1, beam_size, 1, 1).view(-1, *k.shape[1:])
            v = v.unsqueeze(1).repeat(1, beam_size, 1, 1).view(-1, *v.shape[1:])
            cross_kv = (k, v)
        # ⚠ Set self_kv to None to avoid repeatedly broadcasting tokens from the first-step decoder
        expanded.append((None, cross_kv))
    return expanded



def iterative_mask_decode(model, encoder_hidden, n_return_sequences=1, tokenizer=None, mode="confidence", rand_cfg=None):
    """
    Vectorized iterative mask filling decode, completely eliminating the Python loop bottleneck.

    Args:
        model: DIFF_GRM model
        encoder_hidden: Encoder output [B, seq_len, emb_dim]
        n_return_sequences: Number of return sequences (will be overridden by top_k_final)
        tokenizer: Tokenizer object
        mode: "confidence" or "random"
        rand_cfg: Random sampling configuration dictionary

    Returns:
        generated_sequences: [B, top_k_final, n_digit] Generated sequences
    """
    device = encoder_hidden.device
    batch_size = encoder_hidden.size(0)
    n_digit = model.n_digit
    codebook_size = model.codebook_size

    # 🚀 Get vectorized beam search parameters from config (supports split-specific configurations)
    if hasattr(model, 'config') and 'vectorized_beam_search' in model.config:
        beam_config = model.config['vectorized_beam_search']

        # Get current split (defaults to val)
        split = model.config.get("current_split", "val")   # "val" / "test"

        # Check if it is a split-specific configuration (supports three syntax variations)
        if split in beam_config:                           # ← Check split-specific first
            BEAM_ACT = int(beam_config[split]["beam_act"])
            BEAM_MAX = int(beam_config[split]["beam_max"])
        elif isinstance(beam_config.get("beam_act"), dict): # Compatible with another syntax: beam_act itself is a dict
            BEAM_ACT = int(beam_config["beam_act"].get(split,
                                                       beam_config["beam_act"]["val"]))
            BEAM_MAX = int(beam_config["beam_max"].get(split,
                                                       beam_config["beam_max"]["val"]))
        else:                                              # Finally fall back to global
            BEAM_ACT = int(beam_config["beam_act"])
            BEAM_MAX = int(beam_config["beam_max"])

        TOP_K_FINAL = min(int(beam_config['top_k_final']), n_return_sequences)
        # 🚀 Fix: Ensure NEG_INF values are float types to avoid YAML string issues
        NEG_INF_FP32 = float(beam_config['neg_inf_fp32'])
        NEG_INF_FP16 = float(beam_config['neg_inf_fp16'])
        # Ensure beam_act does not exceed beam_max
        assert BEAM_ACT <= BEAM_MAX, "beam_act should not exceed beam_max"
    else:
        # 🚀 Fix: Uniform configuration, no longer has fallback
        raise ValueError("Missing 'vectorized_beam_search' configuration in model.config")

    # ---------- ① Parse beam_size (special handling for random mode) ----------
    if mode == "random":
        # If random_beam specifies beam_act/beam_max, override them
        rb_cfg = model.config.get("random_beam", {})
        BEAM_ACT = int(rb_cfg.get("beam_act", BEAM_ACT))
        BEAM_MAX = int(rb_cfg.get("beam_max", BEAM_MAX))
        # Ensure beam_act does not exceed beam_max
        assert BEAM_ACT <= BEAM_MAX, "random_beam.beam_act should not exceed random_beam.beam_max"

    # ---------- ② Randomize column order once (random mode only) ----------
    decode_order = None
    if mode == "random":
        # 🚀 Fix: Save current random seed state to avoid polluting training
        original_state = torch.get_rng_state()
        try:
            seed = model.config.get("random_beam", {}).get("seed")
            if seed is not None:
                torch.manual_seed(seed)
            decode_order = torch.randperm(n_digit).tolist()      # e.g. [1,5,3,7,0,2,6,4]
            if batch_size == 1:  # Only print for single sample to avoid flooding logs with multi-workers
                print(f"[RANDOM_BEAM] 🎲 Decode order: {decode_order}")
        finally:
            # 🚀 Restore original random seed state
            torch.set_rng_state(original_state)

    # Constants
    MASK_ID = tokenizer.mask_token if tokenizer is not None else -1
    VOC = codebook_size

    # Reduce log noise
    if batch_size == 1:  # Only print for single sample to avoid flooding logs with multi-workers
        print(f"[VECTORIZED_BEAM] 🚀 Using optimized beam search:")
        print(f"[VECTORIZED_BEAM] BEAM_ACT: {BEAM_ACT}, BEAM_MAX: {BEAM_MAX}, TOP_K_FINAL: {TOP_K_FINAL}")

    # Step 0: Full mask prediction, get probabilities for all positions
    with torch.no_grad():
        # Build mask_positions: all 1s mean everything is masked
        mask_positions = torch.ones(batch_size, n_digit, device=device)

        # Build batch
        batch_dict = {
            'decoder_input_ids': torch.zeros(batch_size, n_digit, device=device, dtype=torch.long),
            'encoder_hidden': encoder_hidden,
            'mask_positions': mask_positions
        }

        # Forward pass - enable KV cache to speed up subsequent inference
        outputs = model.forward_decoder_only(batch_dict, digit=None, use_cache=True)
        all_logits = outputs.logits  # [B, n_digit, codebook_size]
        initial_kv_cache = outputs.past_key_values  # Save initial KV cache

        # Calculate log probabilities
        all_log_probs = F.log_softmax(all_logits, dim=-1)  # [B, n_digit, codebook_size]

        if mode == "random":
            # === random mode: only look at the first column ===
            first_col = decode_order[0]
            probs_col = all_log_probs[:, first_col, :]          # [B, VOC]
            top_k_probs, top_k_idx = torch.topk(probs_col, k=BEAM_ACT, dim=-1)  # [B, BEAM_ACT]

            # Parse position and tokens
            first_col_tensor = torch.full((batch_size, BEAM_ACT), first_col, device=device, dtype=torch.long)
            first_token = top_k_idx
        else:
            # === confidence mode: global top-k ===
            # Flatten probabilities of all positions: [B, n_digit * codebook_size]
            flattened_log_probs = all_log_probs.view(batch_size, -1)

            # Take top BEAM_ACT candidates
            top_k_probs, top_k_indices = torch.topk(flattened_log_probs, k=BEAM_ACT)

            # Parse position and tokens
            first_col_tensor = top_k_indices // VOC      # Which digit [B, BEAM_ACT]
            first_token = top_k_indices % VOC     # ID inside the codebook [B, BEAM_ACT]

        # 🚀 Fixed-size beam tensor, allocated once (critical optimization)
        beam_ids = torch.full((batch_size, BEAM_MAX, n_digit), MASK_ID,
                             dtype=torch.long, device=device)

        # Determine NEG_INF value
        NEG_INF = NEG_INF_FP16 if top_k_probs.dtype == torch.float16 else NEG_INF_FP32
        beam_logp = torch.full((batch_size, BEAM_MAX), NEG_INF,
                              dtype=top_k_probs.dtype, device=device)

        # Fill results from the first step
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)  # [B, 1]
        beam_indices = torch.arange(BEAM_ACT, device=device).unsqueeze(0)     # [1, BEAM_ACT]

        beam_ids[batch_indices, beam_indices, first_col_tensor] = first_token
        beam_logp[:, :BEAM_ACT] = top_k_probs

        # 🚀 Fix: Expand to BEAM_MAX to ensure sufficient capacity
        encoder_hidden_expanded = encoder_hidden.unsqueeze(1).repeat(1, BEAM_MAX, 1, 1)
        encoder_hidden_expanded = encoder_hidden_expanded.view(-1, encoder_hidden.size(1), encoder_hidden.size(2))

        # After Step-0 ends, generate a broadcasted cache once for subsequent reuse
        kv_cache_for_act = expand_cross_kv_for_beams(initial_kv_cache, BEAM_ACT)
        kv_cache_final = expand_cross_kv_for_beams(initial_kv_cache, BEAM_ACT)  # Used for final step

    # Steps 1-2: Vectorized beam expansion (completely eliminates Python loops)
    if mode == "random":
        # === random mode: loop according to decode_order ===
        for step, cur_col in enumerate(decode_order[1:], 1):
            with torch.no_grad():
                # Only use the first BEAM_ACT valid beams
                active_beam_ids = beam_ids[:, :BEAM_ACT, :]      # [B, BEAM_ACT, n_digit]
                active_beam_logp = beam_logp[:, :BEAM_ACT]       # [B, BEAM_ACT]

                # Build mask_positions for current state
                mask_positions = (active_beam_ids == MASK_ID).float()  # [B, BEAM_ACT, n_digit]

                # Reshape to decoder input format
                decoder_input = torch.clamp(active_beam_ids, min=0).view(-1, n_digit)  # [B*BEAM_ACT, n_digit]
                mask_pos_flat = mask_positions.view(-1, n_digit)  # [B*BEAM_ACT, n_digit]

                # 🚀 Use pre-generated KV cache to achieve true cache reuse
                expanded_kv_cache = kv_cache_for_act

                # Build batch
                batch_dict = {
                    'decoder_input_ids': decoder_input,
                    'encoder_hidden': encoder_hidden_expanded[:batch_size * BEAM_ACT],  # Only use the first BEAM_ACT part
                    'mask_positions': mask_pos_flat
                }

                # Forward pass
                outputs = model.forward_decoder_only(batch_dict, digit=None,
                                                   past_key_values=expanded_kv_cache, use_cache=True)
                all_logits = outputs.logits  # [B*BEAM_ACT, n_digit, codebook_size]

                # Reshape to beam dimension
                all_logits = all_logits.view(batch_size, BEAM_ACT, n_digit, codebook_size)

                # 🚀 Vectorized mask processing (core optimization)
                all_log_probs = F.log_softmax(all_logits, dim=-1)

                # Only consider masked positions
                mask_expanded = mask_positions.unsqueeze(-1)  # [B, BEAM_ACT, n_digit, 1]
                masked_log_probs = all_log_probs + (1 - mask_expanded) * NEG_INF

                # === random mode: only look at current column ===
                logits = masked_log_probs[:, :, cur_col, :]                     # [B, BEAM_ACT, VOC]

                joint_lp = logits + active_beam_logp.unsqueeze(-1)              # [B, BEAM_ACT, VOC]
                flat_lp  = joint_lp.view(batch_size, -1)                        # [B, BEAM_ACT*VOC]
                best_lp, flat_idx = torch.topk(flat_lp, k=BEAM_ACT)            # ← top-k, no sampling

                # Parse indices
                parent_beam_ids = flat_idx // VOC                               # [B, BEAM_ACT]
                token_ids = flat_idx % VOC                                      # [B, BEAM_ACT]

                # Update beams
                batch_range = torch.arange(batch_size, device=device).unsqueeze(1)  # [B, 1]
                new_beam_ids = active_beam_ids[batch_range, parent_beam_ids]        # [B, BEAM_ACT, n_digit]
                new_beam_ids.scatter_(2, torch.full((batch_size, BEAM_ACT), cur_col, device=device, dtype=torch.long).unsqueeze(-1), token_ids.unsqueeze(-1))

                # Update beam state
                beam_ids[:, :BEAM_ACT, :] = new_beam_ids
                beam_logp[:, :BEAM_ACT] = best_lp

                # Clear invalid beams (maintain BEAM_MAX size)
                if BEAM_ACT < BEAM_MAX:
                    beam_ids[:, BEAM_ACT:, :] = MASK_ID
                    beam_logp[:, BEAM_ACT:] = NEG_INF
    else:
        # === confidence mode: original logic ===
        for step in range(1, n_digit - 1):
            with torch.no_grad():
                # Only use the first BEAM_ACT valid beams
                active_beam_ids = beam_ids[:, :BEAM_ACT, :]      # [B, BEAM_ACT, n_digit]
                active_beam_logp = beam_logp[:, :BEAM_ACT]       # [B, BEAM_ACT]

                # Build mask_positions for current state
                mask_positions = (active_beam_ids == MASK_ID).float()  # [B, BEAM_ACT, n_digit]

                # Reshape to decoder input format
                decoder_input = torch.clamp(active_beam_ids, min=0).view(-1, n_digit)  # [B*BEAM_ACT, n_digit]
                mask_pos_flat = mask_positions.view(-1, n_digit)  # [B*BEAM_ACT, n_digit]

                # 🚀 Use pre-generated KV cache to achieve true cache reuse
                expanded_kv_cache = kv_cache_for_act

                # Build batch
                batch_dict = {
                    'decoder_input_ids': decoder_input,
                    'encoder_hidden': encoder_hidden_expanded[:batch_size * BEAM_ACT],  # Only use the first BEAM_ACT part
                    'mask_positions': mask_pos_flat
                }

                # Forward pass
                outputs = model.forward_decoder_only(batch_dict, digit=None,
                                                   past_key_values=expanded_kv_cache, use_cache=True)
                all_logits = outputs.logits  # [B*BEAM_ACT, n_digit, codebook_size]

                # Reshape to beam dimension
                all_logits = all_logits.view(batch_size, BEAM_ACT, n_digit, codebook_size)

                # 🚀 Vectorized mask processing (core optimization)
                all_log_probs = F.log_softmax(all_logits, dim=-1)

                # Only consider masked positions
                mask_expanded = mask_positions.unsqueeze(-1)  # [B, BEAM_ACT, n_digit, 1]
                masked_log_probs = all_log_probs + (1 - mask_expanded) * NEG_INF

                # Concat all possible candidates: [B, BEAM_ACT, n_digit * codebook_size]
                flattened_log_probs = masked_log_probs.view(batch_size, BEAM_ACT, -1)

                # 🚀 Use unified branch selection logic
                best_logprobs, new_beam_ids = _beam_step_select(
                    mode=mode,
                    logp_matrix=flattened_log_probs,          # [B, act, n_digit*VOC]
                    cur_beam_logp=active_beam_logp,           # [B, act]
                    beam_ids=active_beam_ids,                 # [B, act, n_digit]
                    n_digit=n_digit, VOC=VOC, beam_act=BEAM_ACT,
                    rand_cfg=rand_cfg or {}
                )

                # Update beam state
                beam_ids[:, :BEAM_ACT, :] = new_beam_ids
                beam_logp[:, :BEAM_ACT] = best_logprobs

                # Clear invalid beams (maintain BEAM_MAX size)
                if BEAM_ACT < BEAM_MAX:
                    beam_ids[:, BEAM_ACT:, :] = MASK_ID
                    beam_logp[:, BEAM_ACT:] = NEG_INF

    # Final step: fill the last position and select top-K
    with torch.no_grad():
        if mode == "random":
            # === random mode: all positions are filled through loops already, directly use current result ===
            active_beam_ids = beam_ids[:, :BEAM_ACT, :]
            final_beam_logp = beam_logp[:, :BEAM_ACT]
        else:
            # === confidence mode: need to fill the last position ===
            # Only process the first BEAM_ACT beams
            active_beam_ids = beam_ids[:, :BEAM_ACT, :]
            active_beam_logp = beam_logp[:, :BEAM_ACT]

            # Find the last MASK position for each beam
            mask_positions = (active_beam_ids == MASK_ID).float()

            # Build decoder input
            decoder_input = torch.clamp(active_beam_ids, min=0).view(-1, n_digit)
            mask_pos_flat = mask_positions.view(-1, n_digit)

            # Use pre-generated KV cache for the final step
            final_expanded_kv_cache = kv_cache_final

            batch_dict = {
                'decoder_input_ids': decoder_input,
                'encoder_hidden': encoder_hidden_expanded[:batch_size * BEAM_ACT],  # Only use the first BEAM_ACT part
                'mask_positions': mask_pos_flat
            }

            # Get logits for all positions
            outputs = model.forward_decoder_only(batch_dict, digit=None,
                                               past_key_values=final_expanded_kv_cache, use_cache=True)
            all_logits = outputs.logits  # [B*BEAM_ACT, n_digit, codebook_size]

            # Reshape and compute log probs
            all_logits = all_logits.view(batch_size, BEAM_ACT, n_digit, codebook_size)
            all_log_probs = F.log_softmax(all_logits, dim=-1)

            # Find the last position that needs to be filled for each beam
            last_mask_pos = torch.argmax(mask_positions.float(), dim=-1)  # [B, BEAM_ACT]

            # Select the best token for corresponding position for each beam
            batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, BEAM_ACT)
            beam_idx = torch.arange(BEAM_ACT, device=device).unsqueeze(0).expand(batch_size, -1)

            final_logits = all_log_probs[batch_idx, beam_idx, last_mask_pos]  # [B, BEAM_ACT, codebook_size]
            best_token_logprobs, best_tokens = torch.max(final_logits, dim=-1)  # [B, BEAM_ACT]

            # Update the final token
            active_beam_ids.scatter_(2, last_mask_pos.unsqueeze(-1), best_tokens.unsqueeze(-1))
            final_beam_logp = active_beam_logp + best_token_logprobs

        # 🚀 Flexible deduplication strategy
        dedup_strategy = "simple"  # Defaults to simple deduplication
        if hasattr(model, 'config') and 'dedup_strategy' in model.config:
            dedup_strategy = model.config['dedup_strategy']

        if dedup_strategy == "none":
            # Strategy 1: No deduplication, directly select top-K
            top_logprobs, top_indices = torch.topk(final_beam_logp, k=min(TOP_K_FINAL, BEAM_ACT), dim=-1)
            batch_range = torch.arange(batch_size, device=device).unsqueeze(1)
            final_sequences = active_beam_ids[batch_range, top_indices]  # [B, TOP_K_FINAL, n_digit]

            if batch_size == 1:
                print(f"[VECTORIZED_BEAM] ✅ Generated {final_sequences.shape[1]} sequences (no deduplication)")

        elif dedup_strategy == "simple":
            # Strategy 2: Simple deduplication + legality check (improved method)
            # ① Tokenizer must be passed in
            assert tokenizer is not None, "tokenizer is required for legality check"

            final_sequences = []
            for b in range(batch_size):
                batch_sequences = active_beam_ids[b]  # [BEAM_ACT, n_digit]
                batch_logprobs = final_beam_logp[b]   # [BEAM_ACT]

                # Sort by probability, then apply simple deduplication + legality check
                sorted_indices = torch.argsort(batch_logprobs, descending=True)
                unique_sequences = []

                for idx in sorted_indices:
                    seq = batch_sequences[idx]
                    # --------- New: Legality check ----------
                    is_legal = tokenizer.codebooks_to_item_id(seq.tolist()) is not None
                    if not is_legal:
                        continue  # Skip illegal sequence directly
                    # ------------------------------------
                    is_duplicate = any(torch.equal(seq, existing) for existing in unique_sequences)
                    if not is_duplicate:
                        unique_sequences.append(seq)
                        if len(unique_sequences) >= TOP_K_FINAL:
                            break

                # Fill missing parts (ensure filled sequences are also legal)
                while len(unique_sequences) < TOP_K_FINAL:
                    if unique_sequences:
                        # If legal sequences exist, repeat the last one
                        unique_sequences.append(unique_sequences[-1])
                    else:
                        # If no legal sequences exist, search for a legal one to fill
                        for idx in range(BEAM_ACT):
                            seq = batch_sequences[idx]
                            if tokenizer.codebooks_to_item_id(seq.tolist()) is not None:
                                unique_sequences.append(seq)
                                break
                        # If a legal sequence still cannot be found, use the first one (illegal, but better than crashing)
                        if not unique_sequences:
                            unique_sequences.append(batch_sequences[0])

                batch_final = torch.stack(unique_sequences[:TOP_K_FINAL])
                final_sequences.append(batch_final)

            final_sequences = torch.stack(final_sequences)
            if batch_size == 1:
                print(f"[VECTORIZED_BEAM] ✅ Generated {final_sequences.shape[1]} unique sequences (simple deduplication + legality check)")

        else:  # weighted
            # Strategy 3: Probability-weighted deduplication + legality check (improved method)
            # ① Tokenizer must be passed in
            assert tokenizer is not None, "tokenizer is required for legality check"

            final_sequences = []
            for b in range(batch_size):
                batch_sequences = active_beam_ids[b]  # [BEAM_ACT, n_digit]
                batch_logprobs = final_beam_logp[b]   # [BEAM_ACT]

                # Map sequence to probability, accumulate probabilities of duplicate sequences (only legal sequences considered)
                seq_to_logprob = {}
                for i in range(BEAM_ACT):
                    seq_tuple = tuple(batch_sequences[i].cpu().tolist())
                    # --------- New: Legality check ----------
                    is_legal = tokenizer.codebooks_to_item_id(list(seq_tuple)) is not None
                    if not is_legal:
                        continue  # Skip illegal sequence directly
                    # ------------------------------------
                    if seq_tuple in seq_to_logprob:
                        # Duplicate sequence: accumulate probability using log-sum-exp (more stable)
                        seq_to_logprob[seq_tuple] = torch.logaddexp(
                            seq_to_logprob[seq_tuple],
                            batch_logprobs[i]
                        )
                    else:
                        seq_to_logprob[seq_tuple] = batch_logprobs[i]

                # Sort by accumulated probability
                sorted_items = sorted(seq_to_logprob.items(),
                                    key=lambda x: x[1].item(), reverse=True)

                # Select the top TOP_K_FINAL unique sequences (already sorted by weighted probability)
                unique_sequences = []
                for seq_tuple, _ in sorted_items[:TOP_K_FINAL]:
                    seq_tensor = torch.tensor(seq_tuple, device=device, dtype=torch.long)
                    unique_sequences.append(seq_tensor)

                # Fill missing parts (ensure filled sequences are also legal)
                while len(unique_sequences) < TOP_K_FINAL:
                    if unique_sequences:
                        # If legal sequences exist, repeat the last one
                        unique_sequences.append(unique_sequences[-1])
                    else:
                        # If no legal sequences exist, search for a legal one to fill
                        for idx in range(BEAM_ACT):
                            seq = batch_sequences[idx]
                            if tokenizer.codebooks_to_item_id(seq.tolist()) is not None:
                                unique_sequences.append(seq)
                                break
                        # If a legal sequence still cannot be found, use the first one (illegal, but better than crashing)
                        if not unique_sequences:
                            unique_sequences.append(batch_sequences[0])

                batch_final = torch.stack(unique_sequences[:TOP_K_FINAL])
                final_sequences.append(batch_final)

            final_sequences = torch.stack(final_sequences)
            if batch_size == 1:
                print(f"[VECTORIZED_BEAM] ✅ Generated {final_sequences.shape[1]} unique sequences (probability-weighted deduplication + legality check)")

    # ------- Calculate current batch statistics -------
    if tokenizer is not None:  # No longer limiting to batch_size==1
        # Fix legal rate calculation: use sequence count as denominator instead of token count
        total_seqs = final_sequences.numel() // n_digit
        legal_final = sum(tokenizer.codebooks_to_item_id(seq.tolist()) is not None
                          for seq in final_sequences.view(-1, n_digit))
        final_legal_ratio = legal_final / total_seqs

        # Fix duplicate rate calculation: use the correct formula
        unique_seqs = len({tuple(seq.tolist()) for seq in final_sequences.view(-1, n_digit)})
        duplicate_ratio = 1 - unique_seqs / total_seqs

        # Return stats for evaluator use, instead of directly printing
        return final_sequences, final_legal_ratio, duplicate_ratio
    # --------------------------------

    return final_sequences


def fast_beam_search_for_eval(model, encoder_hidden, beam_size=10, max_len=4, tokenizer=None, mode="confidence", rand_cfg=None):
    """
    Fast vectorized beam search optimized specifically for validation.
    Adopts a strategy consistent with TensorFlow: fixed 512 beams for the first 3 steps, then takes top-K.

    Args:
        model: DIFF_GRM model
        encoder_hidden: Encoder output [batch_size, seq_len, hidden_dim]
        beam_size: Final beam size (will be overridden by TOP_K_FINAL)
        max_len: Maximum generation length
        tokenizer: Tokenizer
        mode: "confidence" or "random"
        rand_cfg: Random sampling configuration dictionary

    Returns:
        torch.Tensor: Generated token sequence [batch_size, TOP_K_FINAL, max_len]
    """
    # Directly call vectorized iterative_mask_decode
    result = iterative_mask_decode(
        model=model,
        encoder_hidden=encoder_hidden,
        n_return_sequences=beam_size,
        tokenizer=tokenizer,
        mode=mode,
        rand_cfg=rand_cfg or {}
    )

    # Handle return value: could be a tuple (sequence + statistics) or just the sequence
    if isinstance(result, tuple):
        return result[0]  # Only return the sequence part
    else:
        return result
