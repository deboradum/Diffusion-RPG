# Adapted from (translated comments)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from genrec.model import AbstractModel
from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer
from .ablate_decode import decode_ablate_confidence


def make_norm(norm_type: str, dim: int, eps: float):
    if (norm_type or "layernorm").lower() == "rmsnorm":
        return nn.RMSNorm(dim, eps=eps)
    return nn.LayerNorm(dim, eps=eps)


class MultiHeadAttention(nn.Module):

    def __init__(self, emb_dim, n_head, attn_drop=0.1, resid_drop=0.1):
        super().__init__()
        assert emb_dim % n_head == 0
        self.n_head = n_head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // n_head

        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.proj = nn.Linear(emb_dim, emb_dim)

        self.attn_dropout = nn.Dropout(attn_drop)
        self.resid_dropout = nn.Dropout(resid_drop)

        # Initialize weights
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x, attention_mask=None, key_value=None, past_key_value=None, use_cache=False, is_decoder_self_attn=False):
        B, T, C = x.size()

        if key_value is not None:
            # Cross attention: Q from x, K,V from key_value
            q = self.qkv(x)[:, :, :self.emb_dim]  # Only take Q part
            k, v = key_value.chunk(2, dim=-1)  # key_value should be [B, T_enc, 2*emb_dim]
            T_kv = k.size(1)
        else:
            # Self attention
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            T_kv = T

        # Handle past key-value cache for incremental decoding
        if past_key_value is not None and use_cache and is_decoder_self_attn:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
            T_kv = k.size(1)

        # Save concatenated full k and v for cache (before reshape)
        k_for_cache = k
        v_for_cache = v

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T_kv, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T_kv, head_dim)
        v = v.view(B, T_kv, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T_kv, head_dim)

        # Scaled dot-product attention
        scale = 1.0 / (self.head_dim ** 0.5)
        att = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, n_head, T, T_kv)

        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: (B, T, T_kv) or (B, 1, T, T_kv)
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)  # Add head dimension
            att = att.masked_fill(attention_mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Apply attention to values
        y = torch.matmul(att, v)  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, emb_dim)

        # Output projection
        y = self.resid_dropout(self.proj(y))

        # Prepare cache for next iteration - save the original 3D k and v
        present_key_value = (k_for_cache, v_for_cache) if use_cache else None

        return y, present_key_value


class FeedForward(nn.Module):

    def __init__(self, emb_dim, n_inner, resid_drop=0.1, act='gelu'):
        super().__init__()
        self.c_fc = nn.Linear(emb_dim, n_inner)
        self.c_proj = nn.Linear(n_inner, emb_dim)
        self.dropout = nn.Dropout(resid_drop)
        self.act = F.gelu if act == 'gelu' else F.relu

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return self.dropout(x)


class EncoderBlock(nn.Module):

    def __init__(self, emb_dim, n_head, n_inner, attn_drop=0.1, resid_drop=0.1,
                 act='gelu', norm_type='layernorm', norm_eps=1e-6):
        super().__init__()
        self.ln_1 = make_norm(norm_type, emb_dim, norm_eps)
        self.attn = MultiHeadAttention(emb_dim, n_head, attn_drop, resid_drop)
        self.ln_2 = make_norm(norm_type, emb_dim, norm_eps)
        self.mlp = FeedForward(emb_dim, n_inner, resid_drop, act)

    def forward(self, x, attention_mask=None):
        # Self-attention + Residual connection (not decoder self-attention)
        attn_output, _ = self.attn(self.ln_1(x), attention_mask=attention_mask, is_decoder_self_attn=False)
        x = x + attn_output

        # Feed-forward network + Residual connection
        x = x + self.mlp(self.ln_2(x))
        return x


class DecoderBlock(nn.Module):

    def __init__(self, emb_dim, n_head, n_inner, attn_drop=0.1, resid_drop=0.1,
                 act='gelu', norm_type='layernorm', norm_eps=1e-6):
        super().__init__()
        self.ln_1 = make_norm(norm_type, emb_dim, norm_eps)
        self.self_attn = MultiHeadAttention(emb_dim, n_head, attn_drop, resid_drop)
        self.ln_2 = make_norm(norm_type, emb_dim, norm_eps)
        self.cross_attn = MultiHeadAttention(emb_dim, n_head, attn_drop, resid_drop)
        self.ln_3 = make_norm(norm_type, emb_dim, norm_eps)
        self.mlp = FeedForward(emb_dim, n_inner, resid_drop, act)

    def forward(self, x, encoder_hidden=None, attention_mask=None,
                past_key_value=None, use_cache=False, cross_key_value=None):
        # Modification: Remove causal mask because the diffusion model does not require strict sequential order
        # Self-attention (without causal mask)
        self_past_kv = None
        cross_past_kv = None
        if past_key_value is not None:
            if len(past_key_value) >= 1:
                self_past_kv = past_key_value[0]
            if len(past_key_value) >= 2:
                cross_past_kv = past_key_value[1]

        attn_output, present_key_value = self.self_attn(
            self.ln_1(x),
            attention_mask=None,  # Do not use causal mask
            past_key_value=self_past_kv,
            use_cache=use_cache,
            is_decoder_self_attn=True
        )
        x = x + attn_output

        # Cross attention
        if encoder_hidden is not None:
            if cross_key_value is not None:
                # 🚀 Use precomputed KV to avoid redundant computation
                encoder_kv = cross_key_value
            else:
                # Legacy compatibility logic: Recompute (only used for non-optimized paths)
                encoder_kv = torch.cat([encoder_hidden, encoder_hidden], dim=-1)  # Concat K and V

            cross_attn_output, cross_present = self.cross_attn(
                self.ln_2(x),
                key_value=encoder_kv,
                past_key_value=cross_past_kv,
                use_cache=use_cache
            )
            x = x + cross_attn_output

            if use_cache:
                present_key_value = (present_key_value, cross_present)

        # Feed-forward network
        x = x + self.mlp(self.ln_3(x))

        return_dict = {}
        return_dict['hidden_states'] = x
        if use_cache:
            return_dict['present_key_value'] = present_key_value

        return return_dict


class ModelOutput:

    def __init__(self):
        self.loss = None
        self.logits = None
        self.hidden_states = None
        self.past_key_values = None


class DIFF_GRMGRAPH(AbstractModel):

    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super().__init__(config, dataset, tokenizer)

        self.config = config
        self.tokenizer = tokenizer
        self.n_digit = config['n_digit']
        self.codebook_size = config['codebook_size']
        self.vocab_size = tokenizer.vocab_size

        # Model dimensions
        self.n_embd = config['n_embd']
        self.n_head = config['n_head']
        self.n_inner = config['n_inner']
        self.dropout = config['dropout']

        # Encoder layers
        self.encoder_n_layer = config['encoder_n_layer']
        self.decoder_n_layer = config['decoder_n_layer']

        # Normalization configuration
        self.norm_type = (config.get('norm_type', 'layernorm') or 'layernorm').lower()
        self.norm_eps  = float(config.get('norm_eps', 1e-6 if self.norm_type=='rmsnorm' else 1e-5))

        # ==== Load new strategy ====
        self.masking_strategy = config.get('masking_strategy', 'random')  # random | sequential

        if self.masking_strategy == 'sequential':
            # Coherent multi-view
            seq_cfg = config.get('sequential_steps', 'auto')
            self.seq_steps = self.n_digit if seq_cfg in (None, 'auto') else int(seq_cfg)
            assert 1 <= self.seq_steps <= self.n_digit, \
                f"sequential_steps must be 1~{self.n_digit}, got {self.seq_steps}"

            # New feature: Multi-path support
            self.sequential_paths = config.get('sequential_paths', 1)
            assert self.sequential_paths >= 1, \
                f"sequential_paths must be >= 1, got {self.sequential_paths}"

            self.augment_factor = self.seq_steps * self.sequential_paths  # Update calculation method
            print(f"[MODEL] ▶ use SEQUENTIAL views: steps={self.seq_steps}, "
                  f"paths={self.sequential_paths}, augment_factor={self.augment_factor}")
            # Remove unnecessary mask_probs setup to save memory
            self.mask_probs = None
        elif self.masking_strategy == 'guided':
            # Confidence-guided coherent multi-view (the model determines the reveal order for each batch)
            guided_cfg = config.get('guided_steps', 'auto')
            self.guided_steps = self.n_digit if guided_cfg in (None, 'auto') else int(guided_cfg)
            # Limit to maximum 4 steps (currently n_digit=4, so exactly 4)
            self.guided_steps = min(self.guided_steps, self.n_digit, 4)
            self.guided_conf_metric = config.get('guided_conf_metric', 'msp')
            assert self.guided_conf_metric in ('msp', 'entropy'), \
                f"guided_conf_metric must be one of ['msp','entropy'], got {self.guided_conf_metric}"
            # New feature: Choose to reveal positions with "most" or "least" confidence
            self.guided_select = config.get('guided_select', 'most')
            assert self.guided_select in ('most', 'least'), \
                f"guided_select must be one of ['most','least'], got {self.guided_select}"
            self.augment_factor = self.guided_steps
            print(f"[MODEL] ▶ GUIDED: steps={self.guided_steps}, metric={self.guided_conf_metric}, "
                  f"select={self.guided_select}, augment_factor={self.augment_factor}")
            self.mask_probs = None
        else:
            # Legacy random masking branch (maintains original logic)
            # Diffusion specific parameters - multi-probability masking configuration
            # New feature: Supports random sampling of a single masking probability within an interval, repeatable via augment_factor
            self.mask_prob_random = bool(config.get('mask_prob_random', False))
            if self.mask_prob_random:
                low = float(config.get('mask_prob_random_min', 0.0))
                high = float(config.get('mask_prob_random_max', 1.0))
                if not (0.0 <= low <= high <= 1.0):
                    raise ValueError(
                        f"mask_prob_random_min/max must satisfy 0.0 <= min <= max <= 1.0, got min={low}, max={high}"
                    )
                sampled_prob = float(np.random.uniform(low, high))
                # As requested: Do not perform multi-view data augmentation when random masking probability is enabled
                self.augment_factor = 1
                self.mask_probs = [sampled_prob]
                self.sampled_mask_prob = sampled_prob
                print(
                    f"[MODEL] Using RANDOMLY-SAMPLED masking prob: {sampled_prob:.4f} (range [{low}, {high}]); disable multi-view (augment_factor=1)"
                )
            elif 'mask_probs' in config and config['mask_probs'] is not None:
                # New approach: Directly specify multiple masking probabilities
                mask_probs_raw = config['mask_probs']

                if isinstance(mask_probs_raw, str):
                    # String format: "1.0,0.75,0.5,0.25"
                    self.mask_probs = [float(p.strip()) for p in mask_probs_raw.split(',')]
                elif isinstance(mask_probs_raw, (list, tuple)):
                    # List or tuple format: [1.0, 0.75, 0.5, 0.25]
                    self.mask_probs = [float(p) for p in mask_probs_raw]
                elif isinstance(mask_probs_raw, (int, float)):
                    # Single numeric value, converted to a single-element list
                    self.mask_probs = [float(mask_probs_raw)]
                else:
                    # Other types, attempt to parse after converting to string
                    try:
                        mask_probs_str = str(mask_probs_raw)
                        self.mask_probs = [float(p.strip()) for p in mask_probs_str.split(',')]
                    except (ValueError, AttributeError):
                        raise ValueError(f"Cannot parse mask_probs: {mask_probs_raw} (type: {type(mask_probs_raw)}). "
                                       "Expected string like '1.0,0.75,0.5,0.25' or list like [1.0, 0.75, 0.5, 0.25]")

                self.augment_factor = len(self.mask_probs)  # Automatically set data augmentation factor
                print(f"[MODEL] Using multi-probability masking: {self.mask_probs}")
            else:
                # Legacy approach: Single masking probability + data augmentation factor
                mask_prob = config.get('mask_prob', 0.5)
                self.augment_factor = config.get('augment_factor', 4)
                self.mask_probs = [float(mask_prob)] * self.augment_factor  # Repeat the same probability
                print(f"[MODEL] Using single-probability masking: {mask_prob} x {self.augment_factor}")

        # Validate the validity of masking probabilities (only valid for the random strategy)
        if self.masking_strategy == 'random' and self.mask_probs is not None:
            for i, prob in enumerate(self.mask_probs):
                if not (0.0 <= prob <= 1.0):
                    raise ValueError(f"mask_probs[{i}] = {prob} is not in valid range [0.0, 1.0]")

        # Embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.n_embd)

        # Add item_mlp aligned with RPG_ED: Compresses n_digit SID tokens into 1 token
        self.item_mlp = nn.Sequential(
            nn.Linear(self.n_digit * self.n_embd, self.n_embd),  # n_digit × d → d
            nn.ReLU(),
            nn.Linear(self.n_embd, self.n_embd)
        )

        # New feature: Mask embedding table, used to represent masked positions
        self.mask_emb_table = nn.Embedding(self.n_digit, self.n_embd)

        # Position embeddings: Only add absolute position embeddings to the encoder (aligned with RPG_ED)
        self.max_history_len = config.get('max_history_len', 50)  # Read from config, defaults to 50
        self.pos_emb_enc = nn.Embedding(self.max_history_len, self.n_embd)
        # Remove decoder position embeddings; decoder only uses mask tokens

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                self.n_embd, self.n_head, self.n_inner,
                config['attn_pdrop'], config['resid_pdrop'],
                act='gelu',
                norm_type=self.norm_type, norm_eps=self.norm_eps
            )
            for _ in range(self.encoder_n_layer)
        ])

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                self.n_embd, self.n_head, self.n_inner,
                config['attn_pdrop'], config['resid_pdrop'],
                act='gelu',
                norm_type=self.norm_type, norm_eps=self.norm_eps
            )
            for _ in range(self.decoder_n_layer)
        ])

        # Layer normalization
        self.ln_f = make_norm(self.norm_type, self.n_embd, self.norm_eps)

        # -- 1.1 Remove old independent heads, change to shared embedding dot-product --
        share_out = self.config.get('share_decoder_output_embedding', True)
        if share_out:
            # Direct weight-tying, no new parameters added
            self.output_adapter = nn.Identity()
            print(f"[DIFF_GRMGRAPH] Using shared embedding dot-product output layer")
        else:
            # Use this line if rollback to independent heads is needed later
            self.output_adapter = nn.Linear(self.n_embd, self.n_embd, bias=False)
            print(f"[DIFF_GRMGRAPH] Using independent Linear output adapter")
        # -------------------------------------------------------------

        # Dropout
        self.drop = nn.Dropout(self.dropout)

        # Initialize weights
        self.apply(self._init_weights)

        # When ablation is enabled, automatically inject confidence_s1/s2/s3 modes to ensure all three run during evaluation phase
        ab_cfg = self.config.get('ablate_decode', {}) or {}
        if bool(ab_cfg.get('enabled', False)):
            modes = list(self.config.get('beam_search_modes', []) or [])
            to_add = ['confidence_s1', 'confidence_s2', 'confidence_s3']
            if 'confidence' in modes:
                base = modes.index('confidence')
                for i, m in enumerate(to_add, 1):
                    if m not in modes:
                        modes.insert(base + i, m)
            else:
                for m in reversed(to_add):
                    if m not in modes:
                        modes.insert(0, m)
            self.config['beam_search_modes'] = modes

        self.generate_w_decoding_graph = False
        self.init_flag = False
        self.chunk_size = config['chunk_size']
        self.num_beams = config['num_beams']
        self.n_edges = config['n_edges']
        self.propagation_steps = config['propagation_steps']

        self.item_id2tokens = self._map_item_tokens().to(self.config['device'])

    def _map_item_tokens(self) -> torch.Tensor:
        """
        Maps item tokens to their corresponding item IDs.

        Returns:
            item_id2tokens (torch.Tensor): A tensor of shape (n_items, n_digit) where each row represents the semantic IDs of an item.
        """
        item_id2tokens = torch.zeros((self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long)
        for k in range(self.tokenizer.n_digit):
            item_id2tokens[:, k] = self.tokenizer.sid_offset + k * self.tokenizer.codebook_size
        for item in self.tokenizer.item2tokens:
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = torch.LongTensor(self.tokenizer.item2tokens[item])
        return item_id2tokens

    def resample_mask_prob_if_needed(self):
        """
        When using random + mask_prob_random=true, this is called at the beginning of each epoch during training
         to resample the masking probability from the designated interval, updating the mask rate and loss scale used for the current epoch.
        """
        if self.masking_strategy == 'random' and getattr(self, 'mask_prob_random', False):
            low = float(getattr(self, 'mask_prob_random_min', 0.0)) if hasattr(self, 'mask_prob_random_min') else float(self.config.get('mask_prob_random_min', 0.0))
            high = float(getattr(self, 'mask_prob_random_max', 1.0)) if hasattr(self, 'mask_prob_random_max') else float(self.config.get('mask_prob_random_max', 1.0))
            if not (0.0 <= low <= high <= 1.0):
                raise ValueError(
                    f"mask_prob_random_min/max must satisfy 0.0 <= min <= max <= 1.0, got min={low}, max={high}"
                )
            sampled_prob = float(np.random.uniform(low, high))
            self.mask_probs = [sampled_prob]  # Single view
            self.sampled_mask_prob = sampled_prob
            print(f"[MODEL] [Epoch-Resample] RANDOM masking prob resampled to {sampled_prob:.4f} (range [{low}, {high}]); augment_factor=1")

    def set_masking_mode(self, strategy: str, **kw):
        """
        Hot-switching during training:
        - strategy: 'guided' | 'sequential' | 'random'
        - kw: Hyperparameters required by the respective strategy (see below)
        """
        self.masking_strategy = strategy

        if strategy == 'sequential':
            # steps
            seq_cfg = kw.get('sequential_steps', self.config.get('sequential_steps', 'auto'))
            self.seq_steps = self.n_digit if seq_cfg in (None, 'auto') else int(seq_cfg)
            self.sequential_paths = int(kw.get('sequential_paths', self.config.get('sequential_paths', 1)))
            self.augment_factor = self.seq_steps * self.sequential_paths
            self.mask_probs = None
            print(f"[SCHEDULE] → SEQUENTIAL: steps={self.seq_steps}, paths={self.sequential_paths}, augment_factor={self.augment_factor}")

        elif strategy == 'guided':
            guided_cfg = kw.get('guided_steps', self.config.get('guided_steps', 'auto'))
            self.guided_steps = self.n_digit if guided_cfg in (None, 'auto') else int(guided_cfg)
            self.guided_steps = min(self.guided_steps, self.n_digit, 4)
            self.guided_conf_metric = kw.get('guided_conf_metric', self.config.get('guided_conf_metric', 'msp'))
            self.guided_select = kw.get('guided_select', self.config.get('guided_select', 'least'))
            # Note: self.config['guided_refresh_each_step'] is read inside forward, so sync it back to config
            self.config['guided_refresh_each_step'] = bool(kw.get(
                'guided_refresh_each_step',
                self.config.get('guided_refresh_each_step', False)
            ))
            self.augment_factor = self.guided_steps
            self.mask_probs = None
            print(f"[SCHEDULE] → GUIDED({self.guided_select}): steps={self.guided_steps}, metric={self.guided_conf_metric}, refresh={self.config['guided_refresh_each_step']}, augment_factor={self.augment_factor}")

        elif strategy == 'random':
            # Retain old logic, overwrite as needed
            self.mask_prob_random = bool(kw.get('mask_prob_random', self.config.get('mask_prob_random', False)))
            if self.mask_prob_random:
                self.mask_probs = [float(np.random.uniform(
                    float(kw.get('mask_prob_random_min', self.config.get('mask_prob_random_min', 0.0))),
                    float(kw.get('mask_prob_random_max', self.config.get('mask_prob_random_max', 1.0)))
                ))]
                self.augment_factor = 1
            else:
                if 'mask_probs' in kw and kw['mask_probs'] is not None:
                    self.mask_probs = [float(p) for p in (kw['mask_probs'] if isinstance(kw['mask_probs'], (list, tuple)) else str(kw['mask_probs']).split(','))]
                    self.augment_factor = len(self.mask_probs)
                else:
                    mp = float(kw.get('mask_prob', self.config.get('mask_prob', 0.5)))
                    af = int(kw.get('augment_factor', self.config.get('augment_factor', 4)))
                    self.mask_probs = [mp] * af
                    self.augment_factor = af
            print(f"[SCHEDULE] → RANDOM: mask_probs={self.mask_probs}, augment_factor={self.augment_factor}")
        else:
            raise ValueError(f"Unknown masking strategy: {strategy}")

    def _compute_digit_logits(self, hidden_last, digit):
        """
        Compute logits using the dot-product of the shared embedding

        Args:
            hidden_last: (B, d_model) - Decoder output hidden states
            digit: 0..n_digit-1 - The digit position to predict

        Returns:
            logits: (B, codebook_size) - Predicted logits
        """
        if digit is None:
            raise ValueError("digit parameter cannot be None, must specify the codebook position to calculate")

        if digit >= self.n_digit:
            raise ValueError(f"digit={digit} out of bounds, should be in [0, {self.n_digit-1}]")

        # 2.1 Take the corresponding slice of the embedding matrix
        # Token ID layout = [PAD, BOS, EOS, digit0 256 items, digit1 256 items, ...]
        start = self.tokenizer.sid_offset + digit * self.codebook_size
        end = start + self.codebook_size  # end exclusive
        # shape: (codebook_size, d_model)
        E_sub = self.embedding.weight[start:end]

        # 2.2 optional adapter
        h = self.output_adapter(hidden_last)  # (B, d_model)

        # 2.3 dot-product yields logits
        # (B, d_model) @ (d_model, codebook_size).T → (B, codebook_size)
        logits = torch.matmul(h, E_sub.t())

        return logits

    @property
    def n_parameters(self) -> str:
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return f"{n_params:,}"

    def forward(self, batch: dict, return_loss=True) -> ModelOutput:
        """
        Diffusion Training: Process masked data and predict masked positions

        Args:
            batch: Dictionary containing the following fields:
                - history_sid: History SID sequence [B, seq_len, n_digit]
                - decoder_input_ids: Decoder inputs [B, n_digit]
                - decoder_labels: True labels [B, n_digit]
        """
        device = next(self.parameters()).device

        # Add debugging information
        if hasattr(self, '_debug_printed'):
            pass
        else:
            print(f"[DIFF_GRMGRAPH] Using RPG_ED-style encoder: MLP compression + fixed 50-length sequence")
            print(f"[DIFF_GRMGRAPH] vocab_size: {self.vocab_size}, codebook_size: {self.codebook_size}")
            print(f"[DIFF_GRMGRAPH] masking_strategy: {self.masking_strategy}")
            if self.masking_strategy == 'random' and self.mask_probs is not None:
                print(f"[DIFF_GRMGRAPH] mask_probs: {self.mask_probs}")
            self._debug_printed = True

        # --- Encoder ---
        history_sid = batch['history_sid'].to(device)  # [B, seq_len, n_digit]
        B, seq_len, n_digit = history_sid.shape

        # Assertion: history_sid should be codebook id (0..K-1) or PAD (-1)
        valid_hist = ((history_sid == -1) | ((history_sid >= 0) & (history_sid < self.codebook_size))).all()
        assert bool(valid_hist), \
            f"history_sid should be codebook id(0..{self.codebook_size-1}) or -1(PAD), but out-of-bounds value found"

        # 1. Convert history SIDs to token IDs
        history_tokens = torch.zeros(B, seq_len, n_digit, dtype=torch.long, device=device)
        for d in range(n_digit):
            # Process PAD: -1 maps to token_id=0(PAD), other codebook_ids add offset normally
            codebook_ids = history_sid[:, :, d]
            token_ids = torch.where(
                codebook_ids == -1,  # PAD position
                torch.zeros_like(codebook_ids),  # Map to token_id=0(PAD)
                codebook_ids + self.tokenizer.sid_offset + d * self.codebook_size  # Normal offset addition
            )
            # Ensure token IDs are within valid range
            token_ids = torch.clamp(token_ids, 0, self.vocab_size - 1)
            history_tokens[:, :, d] = token_ids

        # 2. Get token embeddings
        tok_emb = self.embedding(history_tokens)  # [B, seq_len, n_digit, d]
        B, S, _, d = tok_emb.shape

        # 3. Reshape and compress via MLP: n_digit SID tokens → 1 item token
        item_emb = tok_emb.reshape(B, S, self.n_digit * d)  # [B, S, n_digit*d]
        item_emb = self.item_mlp(item_emb)  # [B, S, d]

        # 4. Add position embeddings (aligned with RPG_ED)
        pos_ids = torch.arange(S, device=item_emb.device)  # (S,)
        pos_emb = self.pos_emb_enc(pos_ids)  # (S, d)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B, S, d)

        # 5. Add position embeddings to item_emb
        encoder_hidden = item_emb + pos_emb  # [B, S, d]
        encoder_hidden = self.drop(encoder_hidden)

        # 6. Handle attention mask for PAD positions
        if 'history_mask' in batch:
            history_mask = batch['history_mask'].to(device)  # [B, seq_len]
            # Create attention mask: True=valid position, False=PAD position
            attention_mask = history_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_len]
            attention_mask = attention_mask.expand(-1, -1, seq_len, -1)  # [B, 1, seq_len, seq_len]
        else:
            attention_mask = None

        # Pass through encoder blocks
        encoder_hidden = encoder_hidden
        for block in self.encoder_blocks:
            encoder_hidden = block(encoder_hidden, attention_mask=attention_mask)

        encoder_hidden = self.ln_f(encoder_hidden)  # [B, seq_len*n_digit, emb_dim]

        # >>> New addition: Zero out encoder_hidden at PAD positions to prevent cross-attn from seeing invalid KV <<<
        if 'history_mask' in batch:
            history_mask = batch['history_mask'].to(device)  # [B, S], True=Valid
            encoder_hidden = encoder_hidden * history_mask.unsqueeze(-1).float()

        if not return_loss:
            # Inference mode, directly return encoder output
            output = ModelOutput()
            output.hidden_states = encoder_hidden
            return output

        # --- Multi-Probability Masking Extension ---
        decoder_input_ids = batch['decoder_input_ids'].to(device)  # [B, n_digit]
        decoder_labels = batch['decoder_labels'].to(device)  # [B, n_digit]

        # Ensure decoder inputs are within valid range
        decoder_input_ids = torch.clamp(decoder_input_ids, 0, self.codebook_size - 1)
        decoder_labels = torch.clamp(decoder_labels, 0, self.codebook_size - 1)

        # ---------- Construct Training Views ----------
        all_masked_input_ids = []
        all_labels = []
        all_mask_positions = []
        all_encoder_hidden = []

        if self.masking_strategy == 'sequential':
            # Coherent multi-view: Supports multi-path parallelism
            for p in range(self.sequential_paths):  # Generate multiple paths first
                # ① Random order for each sample in this path
                orders = torch.argsort(torch.rand(B, self.n_digit, device=device), dim=1)

                # ② step-0: Fully MASKED
                full_mask = torch.ones(B, self.n_digit, dtype=torch.bool, device=device)
                inp0 = decoder_input_ids.new_zeros(B, self.n_digit)        # All 0 → MASK
                all_masked_input_ids.append(inp0)
                all_labels.append(decoder_labels)
                all_mask_positions.append(full_mask.float())
                all_encoder_hidden.append(encoder_hidden)

                # ③ step-1 … step-(seq_steps-1) : Gradually reveal based on random order
                for reveal in range(1, self.seq_steps):        # 1 .. seq_steps-1
                    mask_pos = torch.ones_like(full_mask)      # Start with fully MASKED

                    # orders[:, :reveal] shape (B, reveal)
                    reveal_idx = orders[:, :reveal]            # Columns to reveal for each sample in this step
                    mask_pos.scatter_(1, reveal_idx, 0)        # Set 0 to indicate "unmasked"

                    inp = decoder_input_ids.clone()
                    inp[mask_pos] = 0                          # Write 0 at masked positions

                    all_masked_input_ids.append(inp)
                    all_labels.append(decoder_labels)
                    all_mask_positions.append(mask_pos.float())
                    all_encoder_hidden.append(encoder_hidden)
        elif self.masking_strategy == 'guided':
            B = decoder_labels.size(0)
            device = decoder_labels.device

            def score_with_mask(cur_mask: torch.Tensor):
                # cur_mask: [B, n_digit], True=masked (needs prediction)
                cur_inp = decoder_input_ids.new_zeros(B, self.n_digit)
                cur_inp[~cur_mask] = decoder_labels[~cur_mask]  # Put "true labels" in unmasked positions

                _was_training = self.training
                self.eval()
                with torch.no_grad():
                    if B == 1:  # Only print during single-sample to avoid crowding multiple workers
                        print(f"[GUIDED] scoring: self.training={self.training}")  # This should be False here
                    logits = self.forward_decoder_only(
                        {
                            'decoder_input_ids': cur_inp,
                            'encoder_hidden': encoder_hidden,
                            'mask_positions': cur_mask.float()
                        },
                        return_loss=False, digit=None, use_cache=False
                    ).logits  # [B, n_digit, K]
                if _was_training:
                    self.train()

                # Calculate confidence (consistent with inference)
                probs = F.softmax(logits, dim=-1)
                if self.guided_conf_metric == 'entropy':
                    ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
                    conf = -ent
                else:  # 'msp'
                    conf = probs.max(dim=-1).values  # ★ Max(...).values used here

                return conf  # [B, n_digit]

            refresh = str(self.config.get('guided_refresh_each_step', False)).lower() in ('1','true','yes','y')
            all_masked_input_ids, all_labels, all_mask_positions, all_encoder_hidden = [], [], [], []

            if not refresh:
                # ------- One-time sorting, no refresh -------
                full_mask = torch.ones(B, self.n_digit, dtype=torch.bool, device=device)
                conf = score_with_mask(full_mask)  # Score with full masking to get ranks
                if self.guided_select == 'most':
                    order = torch.argsort(conf, 1, True)
                else:
                    order = torch.argsort(conf, 1, False)

                for t in range(1, self.guided_steps + 1):
                    cur_mask = torch.zeros(B, self.n_digit, dtype=torch.bool, device=device)
                    cols = order[:, :t]
                    cur_mask.scatter_(1, cols, True)

                    cur_inp = decoder_input_ids.new_zeros(B, self.n_digit)
                    cur_inp[~cur_mask] = decoder_labels[~cur_mask]

                    all_masked_input_ids.append(cur_inp)
                    all_labels.append(decoder_labels)
                    all_mask_positions.append(cur_mask.float())
                    all_encoder_hidden.append(encoder_hidden)
            else:
                # ------- Refresh every step -------
                cur_mask = torch.zeros(B, self.n_digit, dtype=torch.bool, device=device)
                for t in range(1, self.guided_steps + 1):
                    conf = score_with_mask(cur_mask)  # Confidence of this step

                    # Columns that have already been masked will not be chosen again
                    if self.guided_select == 'most':
                        conf = conf.masked_fill(cur_mask, -1e9)
                        cols = torch.argmax(conf, dim=1, keepdim=True)  # Select 1 column per sample
                    else:
                        conf = conf.masked_fill(cur_mask,  1e9)
                        cols = torch.argmin(conf, dim=1, keepdim=True)

                    cur_mask.scatter_(1, cols, True)

                    cur_inp = decoder_input_ids.new_zeros(B, self.n_digit)
                    cur_inp[~cur_mask] = decoder_labels[~cur_mask]

                    all_masked_input_ids.append(cur_inp)
                    all_labels.append(decoder_labels)
                    all_mask_positions.append(cur_mask.float())
                    all_encoder_hidden.append(encoder_hidden)

        else:
            # Old random masking branch (maintains original logic)
            # LLaDA style: If mask_prob_random is enabled, sample a mask rate independently per batch
            batch_mask_prob = None
            if getattr(self, 'mask_prob_random', False):
                low = float(self.config.get('mask_prob_random_min', 0.0))
                high = float(self.config.get('mask_prob_random_max', 1.0))
                # Sample using torch to maintain consistency with the global random seed
                batch_mask_prob = float(torch.empty(1).uniform_(low, high).item())
            for view_idx, mask_prob in enumerate(self.mask_probs):
                if batch_mask_prob is not None:
                    mask_prob = batch_mask_prob
                # Generate mask for current masking probability
                mask_positions = torch.rand(B, self.n_digit, device=device) < mask_prob  # [B, n_digit]

                # Ensure at least one position is masked for each sample
                no_mask_samples = ~mask_positions.any(dim=1)  # [B]
                if no_mask_samples.any():
                    # Force mask the first position for samples with no masks
                    mask_positions[no_mask_samples, 0] = True

                # Apply mask: Set masked positions to 0
                masked_input_ids = decoder_input_ids.clone()  # [B, n_digit]
                masked_input_ids[mask_positions] = 0

                # Store data for the current view
                all_masked_input_ids.append(masked_input_ids)
                all_labels.append(decoder_labels)  # Labels remain unchanged
                all_mask_positions.append(mask_positions.float())
                all_encoder_hidden.append(encoder_hidden)  # Each view uses the identical encoder output

        # Merge all views: [B*n_views, ...]
        decoder_input_ids = torch.cat(all_masked_input_ids, dim=0)  # [B*n_views, n_digit]
        decoder_labels = torch.cat(all_labels, dim=0)  # [B*n_views, n_digit]
        mask_positions = torch.cat(all_mask_positions, dim=0)  # [B*n_views, n_digit]
        encoder_hidden = torch.cat(all_encoder_hidden, dim=0)  # [B*n_views, seq_len*n_digit, emb_dim]

        # Update batch size and validate shape
        B_expanded = B * self.augment_factor

        # Shape validation
        assert decoder_input_ids.shape[0] == B_expanded, f"decoder_input_ids shape mismatch: {decoder_input_ids.shape[0]} vs {B_expanded}"
        assert decoder_labels.shape[0] == B_expanded, f"decoder_labels shape mismatch: {decoder_labels.shape[0]} vs {B_expanded}"
        assert mask_positions.shape[0] == B_expanded, f"mask_positions shape mismatch: {mask_positions.shape[0]} vs {B_expanded}"
        assert encoder_hidden.shape[0] == B_expanded, f"encoder_hidden shape mismatch: {encoder_hidden.shape[0]} vs {B_expanded}"

        # Consistency check: Guided strategy should monotonically increase mask count
        if self.masking_strategy == 'guided':
            m = mask_positions.view(B, self.augment_factor, self.n_digit).sum(-1)  # [B, 4]
            assert torch.all(m[:, 1:] >= m[:, :-1]), "guided views should increase masked count monotonically"

        # --- Decoder (Training Mode) ---
        # 🚀 Training stage also uses the cross-attention projection consistent with inference
        encoder_kv_list = []
        for blk in self.decoder_blocks:
            # Execute W_k/W_v projection, fully consistent with inference
            kv_proj = blk.cross_attn.qkv(encoder_hidden)  # [B_expanded, seq_len, 3*emb_dim]
            # Extract K and V components (skip Q component)
            k = kv_proj[..., self.n_embd:2*self.n_embd]  # [B_expanded, seq_len, emb_dim]
            v = kv_proj[..., 2*self.n_embd:]              # [B_expanded, seq_len, emb_dim]
            # Concatenate K and V
            layer_kv = torch.cat([k, v], dim=-1)  # [B_expanded, seq_len, 2*emb_dim]
            encoder_kv_list.append(layer_kv)

        # Build decoder input embeddings
        decoder_emb = torch.zeros(B_expanded, self.n_digit, self.n_embd, device=device)

        for d in range(self.n_digit):
            # Get codebook IDs for current digit
            codebook_ids = decoder_input_ids[:, d]  # [B_expanded]

            # Convert to token IDs, adding safety check
            token_ids = codebook_ids + self.tokenizer.sid_offset + d * self.codebook_size
            token_ids = torch.clamp(token_ids, 0, self.vocab_size - 1)

            # Safe embedding lookup
            token_emb = self.embedding(token_ids)  # [B_expanded, emb_dim]

            # Get mask embedding for current digit
            mask_emb = self.mask_emb_table.weight[d]  # [emb_dim] - No extra tensor creation
            mask_emb = mask_emb.unsqueeze(0).expand(B_expanded, -1)  # [B_expanded, emb_dim]

            # Determine which embedding to use based on mask_positions
            is_masked = mask_positions[:, d].unsqueeze(-1)  # [B_expanded, 1]
            decoder_emb[:, d, :] = torch.where(is_masked.bool(), mask_emb, token_emb)

        # Remove positional encodings: Decoder only uses masks, no positional encoding needed
        decoder_emb = self.drop(decoder_emb)

        # Pass through decoder blocks with consistent cross-attention
        decoder_hidden = decoder_emb
        for i, block in enumerate(self.decoder_blocks):
            block_output = block(
                decoder_hidden,
                encoder_hidden=encoder_hidden,     # Still pass H for fallback convenience
                past_key_value=None,               # Do not use KV cache during training
                use_cache=False,                   # Do not use KV cache during training
                cross_key_value=encoder_kv_list[i] # 🚀 Use precomputed KV, consistent with inference
            )
            decoder_hidden = block_output['hidden_states']

        decoder_hidden = self.ln_f(decoder_hidden)  # [B_expanded, n_digit, emb_dim]

        # Calculate loss
        if self.masking_strategy == 'random' and getattr(self, 'mask_prob_random', False):
            # LLaDA style: Aggregate by masked positions per sample first, then multiply by 1/t.
            # This effectively suppresses scale variance caused by different mask rates.
            # Here t uses the "actual mask rate" instead of sample parameters to avoid excessively
            # large weights when a tiny t is forced to mask a single position.
            per_sample_loss = torch.zeros(B_expanded, device=device)
            for d in range(self.n_digit):
                logits_d = self._compute_digit_logits(decoder_hidden[:, d, :], digit=d)
                labels_d = decoder_labels[:, d]
                mask_d = mask_positions[:, d].float()
                loss_d = F.cross_entropy(
                    logits_d, labels_d, reduction='none',
                    label_smoothing=self.config.get('label_smoothing', 0.1)
                )
                per_sample_loss += loss_d * mask_d  # Only count masked positions
            # Actual mask rate t_i: proportion of masked tokens per sample
            t_actual = mask_positions.float().mean(dim=1)  # [B_expanded]
            t_actual = torch.clamp(t_actual, min=1e-6)
            total_loss = (per_sample_loss / t_actual).mean()  # Average across batch
        else:
            # Original logic: Only calculate loss on masked positions, and average by number of masked tokens
            total_loss = 0.0
            total_weight = 0.0
            for d in range(self.n_digit):
                logits_d = self._compute_digit_logits(decoder_hidden[:, d, :], digit=d)
                labels_d = decoder_labels[:, d]
                mask_d = mask_positions[:, d].float()
                loss_d = F.cross_entropy(
                    logits_d, labels_d, reduction='none',
                    label_smoothing=self.config.get('label_smoothing', 0.1)
                )
                total_loss += (loss_d * mask_d).sum()
                total_weight += mask_d.sum()
            if total_weight > 0:
                total_loss = total_loss / total_weight
            else:
                total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        output = ModelOutput()
        output.loss = total_loss
        output.hidden_states = decoder_hidden
        output.logits = None  # Do not return all logits to save memory

        return output

    def forward_decoder_only(self, batch: dict, return_loss=False, digit=None,
                            past_key_values=None, use_cache=False) -> ModelOutput:
        """
        Runs the decoder part only, used for iterative prediction during inference

        Args:
            batch: Dictionary containing the following fields:
                - decoder_input_ids: Decoder inputs [B, n_digit]
                - encoder_hidden: Encoder outputs [B, seq_len, emb_dim]
                - mask_positions: Mask positions [B, n_digit] (Optional)
            digit: Digit position to predict
            past_key_values: Cached key-value pairs to accelerate inference
            use_cache: Whether to use KV cache
        """
        device = next(self.parameters()).device

        decoder_input_ids = batch['decoder_input_ids'].to(device)  # [B, n_digit]
        encoder_hidden = batch['encoder_hidden'].to(device)  # [B, seq_len, emb_dim]
        B, n_digit = decoder_input_ids.shape

        # Get mask positions; if not provided, assume all positions are unmasked
        if 'mask_positions' in batch:
            mask_positions = batch['mask_positions'].to(device)  # [B, n_digit]
        else:
            mask_positions = torch.zeros(B, n_digit, device=device)

        # 🚀 Cross-KV Cache Optimization: Computed at step 1, reused from past_key_values in subsequent steps
        encoder_kv_list = None

        if past_key_values is None and use_cache:
            # Step 1: Precompute cross-attention KV for each layer
            encoder_kv_list = []
            for blk in self.decoder_blocks:
                with torch.no_grad():
                    kv_proj = blk.cross_attn.qkv(encoder_hidden)  # [B, seq_len, 3*emb_dim]
                    k = kv_proj[..., self.n_embd:2*self.n_embd]  # [B, seq_len, emb_dim]
                    v = kv_proj[..., 2*self.n_embd:]              # [B, seq_len, emb_dim]
                    layer_kv = torch.cat([k, v], dim=-1)  # [B, seq_len, 2*emb_dim]
                encoder_kv_list.append(layer_kv)
        elif past_key_values is not None:
            # Subsequent steps: Extract cross-KV from past_key_values to achieve true cache reuse
            encoder_kv_list = []
            for layer_cache in past_key_values:
                if layer_cache is not None and len(layer_cache) >= 2:
                    _, cross_kv = layer_cache
                    if cross_kv is not None:
                        cross_key, cross_value = cross_kv
                        layer_kv = torch.cat([cross_key, cross_value], dim=-1)
                        encoder_kv_list.append(layer_kv)
                    else:
                        encoder_kv_list.append(None)
                else:
                    encoder_kv_list.append(None)

        # Build decoder input embeddings
        decoder_emb = torch.zeros(B, n_digit, self.n_embd, device=device)

        for d in range(n_digit):
            # Get token IDs for current digit, adding safety check
            token_ids = decoder_input_ids[:, d] + self.tokenizer.sid_offset + d * self.codebook_size
            token_ids = torch.clamp(token_ids, 0, self.vocab_size - 1)
            token_emb = self.embedding(token_ids)  # [B, emb_dim]

            # Get mask embedding for current digit
            mask_emb = self.mask_emb_table.weight[d]  # [emb_dim] - No extra tensor creation
            mask_emb = mask_emb.unsqueeze(0).expand(B, -1)  # [B, emb_dim]

            # Determine which embedding to use based on mask_positions
            is_masked = mask_positions[:, d].unsqueeze(-1)  # [B, 1]
            decoder_emb[:, d, :] = torch.where(is_masked.bool(), mask_emb, token_emb)

        # Remove positional encodings: Decoder only uses masks, no positional encoding needed
        decoder_emb = self.drop(decoder_emb)

        # Pass through decoder blocks with KV cache support
        decoder_hidden = decoder_emb
        present_key_values = []

        for i, block in enumerate(self.decoder_blocks):
            # Get past_key_value for current layer
            layer_past = past_key_values[i] if past_key_values is not None else None

            # 🚀 Pass in precomputed cross-KV to implement cache reuse
            current_cross_kv = encoder_kv_list[i] if encoder_kv_list is not None else None

            block_output = block(
                decoder_hidden,
                encoder_hidden=encoder_hidden,     # Still pass H for fallback convenience
                past_key_value=layer_past,
                use_cache=use_cache,
                cross_key_value=current_cross_kv   # Passed only on the first call
            )
            decoder_hidden = block_output['hidden_states']

            # Collect new key-value cache
            if use_cache:
                layer_present = block_output.get('present_key_value')
                if layer_present is not None and len(layer_present) >= 2:
                    self_present, cross_present = layer_present
                    # Ensure cross_present stores split K and V for the next cache session
                    if cross_present is not None:
                        # cross_present should be in (K, V) format
                        layer_kv = encoder_kv_list[i] if encoder_kv_list is not None else None
                        if layer_kv is not None:
                            k, v = layer_kv.chunk(2, dim=-1)  # Split K and V
                            cross_present = (k, v)  # Save in separated format
                    present_key_values.append((self_present, cross_present))
                else:
                    present_key_values.append(layer_present)

        # If not using cache, set to None
        if not use_cache:
            present_key_values = None

        decoder_hidden = self.ln_f(decoder_hidden)  # [B, n_digit, emb_dim]

        # Calculate logits for specified digit
        if digit is not None:
            logits = self._compute_digit_logits(decoder_hidden[:, digit, :], digit=digit)
        else:
            # If no digit is specified, calculate logits for all positions
            logits = []
            for d in range(n_digit):
                logits_d = self._compute_digit_logits(decoder_hidden[:, d, :], digit=d)
                logits.append(logits_d)
            logits = torch.stack(logits, dim=1)  # [B, n_digit, codebook_size]

        output = ModelOutput()
        output.hidden_states = decoder_hidden
        output.logits = logits
        output.past_key_values = present_key_values

        return output

    def build_ii_sim_mat(self):
        # Assuming n_digit=32, codebook_size=256
        n_items = self.dataset.n_items
        n_digit = self.tokenizer.n_digit
        codebook_size = self.tokenizer.codebook_size
        device = self.embedding.weight.device
        start_idx = self.tokenizer.sid_offset

        # 1) Reshape first 8192 rows of token embeddings into [32, 256, d]
        #    ignoring 2 rows which might be special tokens
        #    shape: (32, 256, d)
        end_idx = start_idx + n_digit * codebook_size
        token_embs = self.embedding.weight[start_idx:end_idx].view(n_digit, codebook_size, -1)

        # 2) Normalize each (256, d) sub-matrix to compute pairwise cosine similarities
        #    We'll do this in a batch for all 32 groups.
        # We do a batch matrix multiply to get (256 x 256) for each group
        # => token_sims: (32, 256, 256)
        token_embs = F.normalize(token_embs, dim=-1, eps=1e-8)
        token_sims = torch.bmm(token_embs, token_embs.transpose(1, 2))

        # 3) Convert [-1, 1] to [0, 1] range
        token_sims_01 = 0.5 * (token_sims + 1.0)  # shape: (32, 256, 256)

        # 4) Prepare an output similarity matrix
        item_item_sim = torch.zeros((n_items, n_items), device=self.embedding.weight.device, dtype=torch.float32)

        # 5) Fill the item-item matrix in chunks
        for i_start in range(1, n_items, self.chunk_size):
            i_end = min(i_start + self.chunk_size, n_items)

            # shape: (chunk_i_size, 32)
            tokens_i = self.item_id2tokens[i_start:i_end]  # sub-block for items i

            for j_start in range(1, n_items, self.chunk_size):
                j_end = min(j_start + self.chunk_size, n_items)

                # shape: (chunk_j_size, 32)
                tokens_j = self.item_id2tokens[j_start:j_end]  # sub-block for items j

                # We want to compute a sub-block of shape: (chunk_i_size, chunk_j_size).
                # For each digit k in [0..31], we look up token_sims_01[k, tokens_i[i, k], tokens_j[j, k]].

                # We'll accumulate the similarity for each of the 32 digits
                block_size_i = i_end - i_start
                block_size_j = j_end - j_start
                sum_block = torch.zeros((block_size_i, block_size_j), device=self.embedding.weight.device, dtype=torch.float32)

                # We'll do a small loop over k=0..31 (which is constant = 32).
                # Each token_sims_01[k] is (256, 256). We gather from it using:
                #   row indices = tokens_i[:, k]
                #   col indices = tokens_j[:, k]
                #
                # The typical approach is:
                #   sub = token_sims_01[k].index_select(0, row_inds).index_select(1, col_inds)
                # Then sum them up across k.
                for k in range(n_digit):
                    # row_inds shape: (block_size_i,)
                    row_inds = tokens_i[:, k] - (start_idx + k * codebook_size)
                    # col_inds shape: (block_size_j,)
                    col_inds = tokens_j[:, k] - (start_idx + k * codebook_size)

                    # token_sims_01[k] -> shape (256, 256)
                    # row-gather => shape (block_size_i, 256)
                    temp = token_sims_01[k].index_select(0, row_inds)
                    # col-gather across dim=1 => shape (block_size_i, block_size_j)
                    temp = temp.index_select(1, col_inds)

                    # Accumulate
                    sum_block += temp

                # Now take the average across the 32 digits
                avg_block = sum_block / n_digit

                # Write back into the final item_item_sim
                item_item_sim[i_start:i_end, j_start:j_end] = avg_block

        return item_item_sim

    def build_adjacency_list(self, item_item_sim):
        return torch.topk(item_item_sim, k=self.n_edges, dim=-1).indices

    def init_graph(self):
        self.tokenizer.log("Building item-item similarity matrix...")
        item_item_sim = self.build_ii_sim_mat()
        self.adjacency = self.build_adjacency_list(item_item_sim)
        self.tokenizer.log("Graph initialized.")

    def graph_propagation(self, token_logits, n_return_sequences):
        batch_size = token_logits.shape[0]
        device = token_logits.device
        start_idx = self.tokenizer.sid_offset
        codebook_size = self.tokenizer.codebook_size

        # Initialize visited nodes tracking
        visited_nodes = {}
        for batch_id in range(batch_size):
            visited_nodes[batch_id] = set()

        # Randomly sample num_beams distinct node IDs in [1..n_nodes]
        topk_nodes_sorted = torch.randint(
            1, self.dataset.n_items,
            (batch_size, self.num_beams),
            dtype=torch.long,
            device=token_logits.device
        )

        # Add initial nodes to visited set
        for batch_id in range(batch_size):
            for node in topk_nodes_sorted[batch_id].cpu().numpy().tolist():
                visited_nodes[batch_id].add(node)

        for sid in range(self.propagation_steps):
            # Find neighbors of these top num_beams nodes
            #      adjacency_list is 0-based internally => need node_id-1
            all_neighbors = self.adjacency[topk_nodes_sorted].view(batch_size, -1)

            next_nodes = []
            for batch_id in range(batch_size):
                neighbors_in_batch = torch.unique(all_neighbors[batch_id])

                # Add neighbors to visited set
                for node in neighbors_in_batch.cpu().numpy().tolist():
                    visited_nodes[batch_id].add(node)

                n_tokens = self.item_id2tokens[neighbors_in_batch]

                # Compute scores by aggregating across digits
                scores = torch.zeros(neighbors_in_batch.shape[0], device=device)
                for k in range(self.n_digit):
                    local_codes = n_tokens[:, k] - (start_idx + k * codebook_size)
                    # Gather logits for this specific digit
                    scores += token_logits[batch_id, k, local_codes]

                scores = scores / self.n_digit
                idxs = torch.topk(scores, self.num_beams).indices
                next_nodes.append(neighbors_in_batch[idxs])
            topk_nodes_sorted = torch.stack(next_nodes, dim=0)

        # Convert visited counts to tensor
        visited_counts = torch.FloatTensor([[len(visited_nodes[batch_id])] for batch_id in range(batch_size)])

        return topk_nodes_sorted[:,:n_return_sequences].unsqueeze(-1), visited_counts


    def generate(self, batch, n_return_sequences=1, mode="graph"):
        """
        Performs inference generation using vectorized iterative mask filling

        Args:
            batch: Batch data containing encoder inputs
            n_return_sequences: Number of return sequences
            mode: "confidence" or "random"

        Returns:
            generated_sequences: [B, top_k_final, n_digit]
        """
        # 🚀 Ensure eval mode is used during inference to turn off dropout
        was_training = self.training
        self.eval()

        try:
            # Get encoder output
            with torch.no_grad():
                encoder_outputs = self.forward(batch, return_loss=False)
                encoder_hidden = encoder_outputs.hidden_states

                if mode == "graph":
                    device = encoder_hidden.device
                    batch_size = encoder_hidden.size(0)
                    n_digit = self.n_digit
                    codebook_size = self.codebook_size

                    denoise_steps = self.config['denoise_inference_steps']
                    steps = min(denoise_steps, n_digit)

                    current_targets = torch.zeros(batch_size, n_digit, dtype=torch.long, device=device)
                    current_mask = torch.ones(batch_size, n_digit, dtype=torch.bool, device=device)

                    final_logits = None

                    for step in range(1, steps + 1):
                        is_masked = current_mask

                        # Build batch for the current step
                        batch_dict = {
                            'decoder_input_ids': current_targets,
                            'encoder_hidden': encoder_hidden,
                            'mask_positions': current_mask.float()
                        }

                        # use_cache=False since self-attention pattern changed
                        outputs = self.forward_decoder_only(batch_dict, digit=None, use_cache=False)
                        logits = outputs.logits  # [B, n_digit, codebook_size]

                        if step == steps:
                            final_logits = logits

                        probs = torch.softmax(logits, dim=-1)
                        max_probs, pred_ids = probs.max(dim=-1)

                        confidence = max_probs.clone()
                        confidence[~is_masked] = 1e9

                        if step == steps:
                            # If it's the final step munmask everything that is left. 0 tokens remain masked.
                            num_to_mask = 0
                        else:
                            # Else only unmask one (n_digit=4 & step=1 = 3 masked.)
                            num_to_mask = self.n_digit - step

                        next_targets = torch.where(is_masked, pred_ids, current_targets)
                        next_mask = torch.zeros_like(current_mask)

                        if num_to_mask > 0:
                            # Keep the `num_to_mask` slots with the lowest confidence masked
                            mask_idx = torch.topk(confidence, k=num_to_mask, dim=-1, largest=False).indices
                            next_mask.scatter_(1, mask_idx, True)

                        current_targets = next_targets
                        current_mask = next_mask

                    token_logits = F.log_softmax(final_logits, dim=-1)
                    # token_logits = final_logits  # [B, n_digit, codebook_size]

                    if not getattr(self, 'init_flag', False):
                        self.init_graph()
                        self.init_flag = True

                    # Run graph propagation
                    if self.generate_w_decoding_graph:
                        if not self.init_flag:
                            self.init_graph()
                            self.init_flag = True
                        preds, _ = self.graph_propagation(
                            token_logits=token_logits,
                            n_return_sequences=n_return_sequences
                        )
                        preds = preds.squeeze(-1)  # [B, maxk]
                    else:
                        start_idx = self.tokenizer.sid_offset
                        item_scores = torch.zeros(batch_size, self.dataset.n_items, device=device)

                        for k in range(n_digit):
                            # Extract the local codebook index (0 to 255)
                            local_codes = self.item_id2tokens[:, k] - (start_idx + k * codebook_size)
                            # local_codes handles all items at once. token_logits is [B, n_digit, codebook_size]
                            item_scores += token_logits[:, k, local_codes]

                        item_scores = item_scores / n_digit
                        item_scores[:, 0] = -float('inf')  # Penalize the PAD item so it doesn't get recommended

                        preds = item_scores.topk(n_return_sequences, dim=-1).indices

                    sid_preds = self.item_id2tokens[preds].clone()  # [B, maxk, n_digit]

                    # Strip the offsets to get pure 0-255 codebook IDs for the evaluator
                    for k in range(n_digit):
                        sid_preds[:, :, k] -= (self.tokenizer.sid_offset + k * codebook_size)

                    return sid_preds

                # Routing: Original 4-step / Random
                if mode in ("confidence", "random"):
                    from .beam import fast_beam_search_for_eval
                    # Routing: Original 4-step / Random
                    generated_sequences = fast_beam_search_for_eval(
                        model=self,
                        encoder_hidden=encoder_hidden,
                        beam_size=n_return_sequences,
                        max_len=self.n_digit,
                        tokenizer=self.tokenizer,
                        mode=mode,
                        rand_cfg=self.config.get("random_beam", {})
                    )
                    return generated_sequences

                # Routing: Ablative 1/2/3 steps (Confidence only)
                if mode.startswith("confidence_s") and bool(self.config.get('ablate_decode', {}).get('enabled', False)):
                    try:
                        steps = int(mode.split("confidence_s")[-1])
                    except Exception:
                        steps = int(self.config.get('ablate_decode', {}).get('steps_default', 3))
                    if steps >= 4:
                        # Fallback to the original 4-step process
                        generated_sequences = fast_beam_search_for_eval(
                            model=self,
                            encoder_hidden=encoder_hidden,
                            beam_size=n_return_sequences,
                            max_len=self.n_digit,
                            tokenizer=self.tokenizer,
                            mode="confidence",
                            rand_cfg=self.config.get("random_beam", {})
                        )
                    else:
                        generated_sequences = decode_ablate_confidence(
                            model=self,
                            encoder_hidden=encoder_hidden,
                            tokenizer=self.tokenizer,
                            steps=steps,
                            n_return_sequences=n_return_sequences,
                        )
                    return generated_sequences
        finally:
            # Restore original training state to prevent affecting subsequent training
            if was_training:
                self.train()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            # LN has bias; RMSNorm only has weight (no bias)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if hasattr(module, "weight") and module.weight is not None:
                torch.nn.init.ones_(module.weight)
        # Note: If output_adapter is Identity(), it requires no initialization
