# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import json
import pickle
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class DIFF_GRMTokenizer(AbstractTokenizer):
    """
    DIFF_GRM Tokenizer for Diffusion-based Generative Recommendation Model

    Special tokens:
    - PAD=0, BOS=1, EOS=2, SID_OFFSET=3

    SID Configuration:
    - n_digit: configurable (e.g., 4, 8, 12), codebook_size=256
    - vocab_size = 3 + n_digit * codebook_size
    """
    def __init__(self, config: dict, dataset: AbstractDataset):
        # Fallback to avoid KeyError
        config.setdefault('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        config.setdefault('num_proc', 1)

        self.n_codebook_bits = self._get_codebook_bits(config['codebook_size'])

        # Choose quantizer: opq_pq (default) | rq_kmeans | none (random)
        self.sid_quantizer = config.get('sid_quantizer', 'opq_pq')
        assert self.sid_quantizer in ('opq_pq', 'rq_kmeans', 'none'), \
            f"sid_quantizer must be one of ['opq_pq','rq_kmeans','none'], got {self.sid_quantizer}"

        # 🚀 Compatibility with legacy configurations: only use disable_opq/index_factory in opq_pq mode
        if self.sid_quantizer == 'opq_pq':
            use_opq = not config.get('disable_opq', False)
            if use_opq:
                self.index_factory = f'OPQ{config["n_digit"]},IVF1,PQ{config["n_digit"]}x{self.n_codebook_bits}'
            else:
                self.index_factory = f'IVF1,PQ{config["n_digit"]}x{self.n_codebook_bits}'
        elif self.sid_quantizer == 'rq_kmeans':
            self.index_factory = f'RQKMEANS{config["n_digit"]}x{self.n_codebook_bits}'
        else:  # 'none'
            self.index_factory = f'RAND{config["n_digit"]}x{self.n_codebook_bits}'

        # Initialize the parent class first to ensure fields like self.config / self.logger are available
        super(DIFF_GRMTokenizer, self).__init__(config, dataset)

        # Write log statements now
        self.log(f'[TOKENIZER] Index factory: {self.index_factory}')
        self.dataset = dataset  # Add dataset reference
        self.item2id = dataset.item2id
        self.id2item = dataset.id_mapping['id2item']

        # Special tokens - Simplify token ID assignment
        self.pad_token = 0
        self.bos_token = 1
        self.eos_token = 2
        self.mask_token = -1  # MASK token is used for inference and is not in the vocabulary
        self.sid_offset = 3  # SID tokens start from 3

        self.item2tokens = self._init_tokenizer(dataset)

        # Create reverse mapping for inference (if it hasn't been created yet)
        if not hasattr(self, 'tokens2item'):
            self.tokens2item = self._create_reverse_mapping()

        # Set collate functions
        from genrec.models.DIFF_GRM.collate import collate_fn_train, collate_fn_val, collate_fn_test
        self.collate_fn = {
            'train': collate_fn_train,
            'val': collate_fn_val,
            'test': collate_fn_test
        }

    @property
    def n_digit(self):
        return self.config['n_digit']

    @property
    def codebook_size(self):
        return self.config['codebook_size']

    @property
    def max_token_seq_len(self) -> int:
        return 1 + self.n_digit  # [BOS] + n_digit SID tokens

    @property
    def vocab_size(self) -> int:
        return 3 + self.n_digit * self.codebook_size  # PAD(0) + BOS(1) + EOS(2) + SID tokens

    def _get_codebook_bits(self, n_codebook):
        x = math.log2(n_codebook)
        assert x.is_integer() and x >= 0, "Invalid value for n_codebook"
        return int(x)

    def _encode_sent_emb(self, dataset: AbstractDataset, output_path: str):
        """Encode sentence embeddings: Supports any Hugging Face SentenceTransformer model id, and applies vector normalization"""
        assert self.config['metadata'] == 'sentence', \
            'DIFF_GRMTokenizer only supports sentence metadata.'

        meta_sentences = []
        for i in range(1, dataset.n_items):
            meta_sentences.append(dataset.item2meta[dataset.id_mapping['id2item'][i]])

        # Accepts any HF model id (e.g., Alibaba-NLP/gte-large-en-v1.5 or BAAI/bge-large-en-v1.5)
        model_id = self.config['sent_emb_model']
        sent_emb_model = SentenceTransformer(model_id, trust_remote_code=True).to(self.config['device'])

        # Encode directly (no prefix needed for GTE/BGE) and apply L2 normalization
        sent_embs = sent_emb_model.encode(
            meta_sentences,
            convert_to_numpy=True,
            batch_size=self.config['sent_emb_batch_size'],
            show_progress_bar=True,
            device=self.config['device'],
            normalize_embeddings=True,
        )

        # Save to disk using the model's basename to avoid conflicts between different models
        sent_embs.tofile(output_path)
        return sent_embs

    def _get_items_for_training(self, dataset: AbstractDataset) -> np.ndarray:
        """Get items used for training"""
        items_for_training = set()

        # Trigger dataset split first (if it hasn't been split yet)
        split_data = dataset.split()

        # Collect all items from the training set
        if 'train' in split_data:
            train_dataset = split_data['train']
            # train_dataset is a Hugging Face Dataset object
            if hasattr(train_dataset, 'column_names') and 'item_seq' in train_dataset.column_names:
                # Iterate through all item_seq
                for item_seq in train_dataset['item_seq']:
                    if isinstance(item_seq, (list, tuple)):
                        items_for_training.update(item_seq)
                    else:
                        items_for_training.add(item_seq)

        # Fix: Ensure mask size matches sent_embs
        # sent_embs only contains items with item_id from 1 to n_items-1
        n_sent_embs = dataset.n_items - 1  # Matches the range(1, dataset.n_items) in _encode_sent_emb
        self.log(f'[TOKENIZER] Items for training: {len(items_for_training)} of {n_sent_embs}')
        self.log(f'[TOKENIZER] Training items sample: {list(items_for_training)[:10]}')

        mask = np.zeros(n_sent_embs, dtype=bool)
        for item in items_for_training:
            item_id = dataset.item2id[item]
            if 1 <= item_id < dataset.n_items:  # Ensure item_id is within valid range
                mask[item_id - 1] = True  # Convert to 0-based index

        self.log(f'[TOKENIZER] Mask shape: {mask.shape}, True count: {np.sum(mask)}')
        return mask

    def _generate_semantic_id_opq(self, sent_embs, sem_ids_path, train_mask):
        """Use OPQ/PQ to generate semantic IDs (compatible with disable_opq), and align using ids from invlists."""
        import faiss

        # Debugging information
        self.log(f'[TOKENIZER] sent_embs shape: {sent_embs.shape}')
        self.log(f'[TOKENIZER] train_mask shape: {train_mask.shape}')
        self.log(f'[TOKENIZER] train_mask True count: {np.sum(train_mask)}')

        # Build index
        if self.config['opq_use_gpu']:
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024 * 1024 * 512)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = self.n_digit >= 56
        faiss.omp_set_num_threads(self.config['faiss_omp_num_threads'])
        index = faiss.index_factory(
            sent_embs.shape[1],
            self.index_factory,
            faiss.METRIC_INNER_PRODUCT
        )
        self.log(f'[TOKENIZER] Training index...')
        if self.config['opq_use_gpu']:
            index = faiss.index_cpu_to_gpu(res, self.config['opq_gpu_id'], index, co)
        index.train(sent_embs[train_mask])
        index.add(sent_embs)
        if self.config['opq_use_gpu']:
            index = faiss.index_gpu_to_cpu(index)

        # Compatible with IndexPreTransform and non-PreTransform
        if isinstance(index, faiss.IndexPreTransform):
            ivf_index = faiss.downcast_index(index.index)
        else:
            ivf_index = faiss.downcast_index(index)

        invlists = faiss.extract_index_ivf(ivf_index).invlists
        ls = invlists.list_size(0)
        # Extract codes and ids, keeping them ordered and aligned
        codes_ptr = invlists.get_codes(0)
        ids_ptr = invlists.get_ids(0)
        pq_codes_u8 = faiss.rev_swig_ptr(codes_ptr, ls * invlists.code_size)
        ids = faiss.rev_swig_ptr(ids_ptr, ls).copy()
        pq_codes_u8 = pq_codes_u8.reshape(-1, invlists.code_size)

        # Parse PQ Code
        faiss_sem_ids = []
        n_bytes = invlists.code_size
        for u8code in pq_codes_u8:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
            code = []
            for _ in range(self.n_digit):
                code.append(bs.read(self.n_codebook_bits))
            faiss_sem_ids.append(code)

        # Align item order using ids
        item2sem_ids = {}
        for pos, iid0 in enumerate(ids):
            item = self.id2item[int(iid0) + 1]
            item2sem_ids[item] = tuple(int(v) for v in faiss_sem_ids[pos])

        self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
        os.makedirs(os.path.dirname(sem_ids_path), exist_ok=True)
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)

    def _generate_semantic_id_random(self, sem_ids_path, n_items, seed=12345):
        """Randomly generate n_digit codebook IDs for each item (uniformly distributed in [0, K-1])."""
        rng = np.random.default_rng(seed)
        item2sem_ids = {}
        for i in range(1, n_items):
            item = self.id2item[i]
            codes = rng.integers(low=0, high=self.codebook_size, size=self.n_digit, endpoint=False, dtype=np.int64)
            item2sem_ids[item] = tuple(int(c) for c in codes.tolist())
        os.makedirs(os.path.dirname(sem_ids_path), exist_ok=True)
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)

    def _generate_semantic_id_rq_kmeans(self, sent_embs, sem_ids_path, train_mask):
        """Use Residual Quantization (KMeans) to generate semantic IDs."""
        import faiss
        d = sent_embs.shape[1]
        K = self.codebook_size
        niter = int(self.config.get('rq_kmeans_niters', 20))
        seed = int(self.config.get('rq_kmeans_seed', 1234))

        # Initialize residuals to original vectors
        residuals = sent_embs.copy().astype(np.float32, copy=False)
        codes_all = np.zeros((sent_embs.shape[0], self.n_digit), dtype=np.int64)

        for stage in range(self.n_digit):
            kmeans = faiss.Kmeans(d=d, k=K, niter=niter, verbose=False, seed=seed + stage)
            kmeans.train(residuals[train_mask])
            # In current Faiss Python, Kmeans.centroids is already a numpy array
            centroids = np.asarray(kmeans.centroids, dtype=np.float32)
            if centroids.ndim == 1:
                centroids = centroids.reshape(K, d)
            elif centroids.shape == (d, K):
                centroids = centroids.T
            assert centroids.shape == (K, d), f"centroids shape {centroids.shape} != {(K, d)}"

            # Assign nearest centroids for all samples
            index = faiss.IndexFlatL2(d)
            index.add(centroids)
            D, I = index.search(residuals, 1)  # I: [N, 1]
            codes_all[:, stage] = I[:, 0].astype(np.int64)

            # Update residuals
            residuals = residuals - centroids[I[:, 0]]

        # Convert to dict
        item2sem_ids = {}
        for i in range(codes_all.shape[0]):
            item = self.id2item[i + 1]
            item2sem_ids[item] = tuple(int(v) for v in codes_all[i].tolist())
        os.makedirs(os.path.dirname(sem_ids_path), exist_ok=True)
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)

    def _sem_ids_to_tokens(self, item2sem_ids: dict) -> dict:
        """Convert semantic IDs to tokens"""
        for item in item2sem_ids:
            tokens = list(item2sem_ids[item])
            # Fix: Reintroduce offset to avoid collision with PAD/BOS
            # Add the corresponding offset to the codebook ID of each digit
            tokens = [t + self.sid_offset + d * self.codebook_size
                     for d, t in enumerate(tokens)]
            item2sem_ids[item] = tuple(tokens)
        return item2sem_ids

    def _init_tokenizer(self, dataset: AbstractDataset):
        """Initialize tokenizer"""
        # Build path - Fix: Use class name and category
        dataset_name = dataset.__class__.__name__  # Use class name, e.g., "AmazonReviews2014"

        # Include category in path if the category attribute exists
        if hasattr(dataset, 'category') and dataset.category:
            cache_dir = os.path.join(
                dataset.cache_dir, 'processed'
            )
        else:
            cache_dir = os.path.join(
                'data', dataset_name, 'processed'
            )

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        # Load semantic IDs (include PCA dimension and quantizer tag in filename to prevent configuration conflicts)
        model_basename = os.path.basename(self.config["sent_emb_model"])
        quant_tag = self.index_factory
        if self.sid_quantizer == 'rq_kmeans':
            quant_tag += f'_seed{self.config.get("rq_kmeans_seed",1234)}_it{self.config.get("rq_kmeans_niters",20)}'
        elif self.sid_quantizer == 'none':
            quant_tag += f'_seed{self.config.get("sid_random_seed",12345)}'
        sem_ids_path = os.path.join(
            cache_dir,
            f'{model_basename}_pca{self.config["sent_emb_pca"]}_{quant_tag}.sem_ids'
        )

        # 🚀 New addition: Check if forced regeneration of quantization results is needed
        force_regenerate = self.config.get('force_regenerate_opq', False)

        # Two embedding files: raw and pca versions, to avoid naming ambiguity and conflicts
        model_basename = os.path.basename(self.config["sent_emb_model"])
        raw_path = os.path.join(
            cache_dir,
            f'{model_basename}_raw_d{self.config["sent_emb_dim"]}.sent_emb'
        )
        pca_path = os.path.join(
            cache_dir,
            f'{model_basename}_pca{self.config["sent_emb_pca"]}.sent_emb'
        )

        # Prepare sentence embeddings if the quantizer requires them; not needed for 'none' mode
        sent_embs = None
        if self.sid_quantizer == 'opq_pq':
            # opq_pq: PCA is allowed (retaining original logic)
            if self.config['sent_emb_pca'] > 0 and os.path.exists(pca_path):
                self.log(f'[TOKENIZER] Loading PCA-ed sentence embeddings from {pca_path}...')
                sent_embs = np.fromfile(pca_path, dtype=np.float32).reshape(
                    -1, self.config['sent_emb_pca']
                )
            elif os.path.exists(raw_path):
                self.log(f'[TOKENIZER] Loading RAW sentence embeddings from {raw_path}...')
                raw_embs = np.fromfile(raw_path, dtype=np.float32).reshape(
                    -1, self.config['sent_emb_dim']
                )
                if self.config['sent_emb_pca'] > 0:
                    self.log(f'[TOKENIZER] Applying PCA to sentence embeddings...')
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
                    training_item_mask = self._get_items_for_training(dataset)
                    pca.fit(raw_embs[training_item_mask])
                    sent_embs = pca.transform(raw_embs)
                    sent_embs = sent_embs.astype(np.float32, copy=False)
                    if self.config.get('normalize_after_pca', True):
                        norms = np.linalg.norm(sent_embs, axis=1, keepdims=True) + 1e-12
                        sent_embs = sent_embs / norms
                    sent_embs.tofile(pca_path)
                else:
                    sent_embs = raw_embs
            else:
                self.log(f'[TOKENIZER] Encoding sentence embeddings...')
                raw_embs = self._encode_sent_emb(dataset, raw_path)
                if self.config['sent_emb_pca'] > 0:
                    self.log(f'[TOKENIZER] Applying PCA to sentence embeddings...')
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
                    training_item_mask = self._get_items_for_training(dataset)
                    pca.fit(raw_embs[training_item_mask])
                    sent_embs = pca.transform(raw_embs)
                    sent_embs = sent_embs.astype(np.float32, copy=False)
                    if self.config.get('normalize_after_pca', True):
                        norms = np.linalg.norm(sent_embs, axis=1, keepdims=True) + 1e-12
                        sent_embs = sent_embs / norms
                    sent_embs.tofile(pca_path)
                else:
                    sent_embs = raw_embs
            self.log(f'[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}')
        elif self.sid_quantizer == 'rq_kmeans':
            # rq_kmeans: Skip PCA and use RAW directly (meets your requirements)
            if os.path.exists(raw_path):
                self.log(f'[TOKENIZER] Loading RAW sentence embeddings from {raw_path}...')
                sent_embs = np.fromfile(raw_path, dtype=np.float32).reshape(
                    -1, self.config['sent_emb_dim']
                )
            else:
                self.log(f'[TOKENIZER] Encoding sentence embeddings (RAW, no PCA for RQ-KMeans)...')
                sent_embs = self._encode_sent_emb(dataset, raw_path)
            self.log(f'[TOKENIZER] Sentence embeddings shape (RAW): {sent_embs.shape}')

        # 🚀 Generate or load quantization results
        if force_regenerate or not os.path.exists(sem_ids_path):
            if force_regenerate:
                self.log(f'[TOKENIZER] Force regenerating quantization results ({self.sid_quantizer})...')
            else:
                self.log(f'[TOKENIZER] Quantization results not found, generating ({self.sid_quantizer})...')
            training_item_mask = self._get_items_for_training(dataset)
            if self.sid_quantizer == 'opq_pq':
                self._generate_semantic_id_opq(sent_embs, sem_ids_path, training_item_mask)
            elif self.sid_quantizer == 'rq_kmeans':
                self._generate_semantic_id_rq_kmeans(sent_embs, sem_ids_path, training_item_mask)
            else:  # 'none'
                self._generate_semantic_id_random(
                    sem_ids_path, n_items=self.dataset.n_items,
                    seed=int(self.config.get('sid_random_seed', 12345))
                )
        else:
            self.log(f'[TOKENIZER] Using existing quantization results from {sem_ids_path}')

        self.log(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')
        item2sem_ids = json.load(open(sem_ids_path, 'r'))
        item2tokens = self._sem_ids_to_tokens(item2sem_ids)

        # 🚀 Mapping filename: reuse the previously constructed quant_tag
        map_tag = f'{model_basename}_pca{self.config["sent_emb_pca"]}_{quant_tag}_{self.n_digit}d'
        fwd_path = os.path.join(cache_dir, f'item_id2tokens_{map_tag}.npy')
        inv_path = os.path.join(cache_dir, f'tokens2item_{map_tag}.pkl')

        # 🚀 Fix ①: Handle consistency of mapping files
        if force_regenerate:
            # When forced to regenerate, directly ignore old files and let the logic below handle "resaving"
            fwd_exists = inv_exists = False
            self.log(f'[TOKENIZER] Force regenerate enabled, ignoring existing mapping files')
        else:
            fwd_exists = os.path.exists(fwd_path)
            inv_exists = os.path.exists(inv_path)

        if fwd_exists and inv_exists:
            # ---------- ① Files already exist ----------
            self.log(f'[TOKENIZER] Loading existing mappings for tag: {map_tag} from {fwd_path}')

            # Reconstruct item2tokens mapping
            item_id2tokens = np.load(fwd_path)
            item2tokens = {}
            for iid, toks in enumerate(item_id2tokens):
                if iid == 0:  # PAD row is all zeros, skip it
                    continue
                item2tokens[self.id2item[iid]] = tuple(toks.tolist())

            # Load inverted index
            with open(inv_path, 'rb') as f:
                self.tokens2item = pickle.load(f)

            self.log(f'[TOKENIZER] Successfully loaded {len(item2tokens)} item mappings')
        else:
            # ---------- ② Files do not exist or force regenerate is triggered, new ones need to be generated ----------
            if force_regenerate:
                self.log(f'[TOKENIZER] Force regenerate enabled, generating new mappings')
            else:
                self.log(f'[TOKENIZER] No existing mappings found for {self.n_digit}-digit, will generate new ones')

            # Regardless of whether files are missing or forceRegenerate is true, save according to the new item2tokens
            self.item2tokens = item2tokens
            self.tokens2item = self._create_reverse_mapping()
            self._save_mappings()  # Only write to disk during "fresh creation"

        # ---- ③ Unification: Attach mapping to instance attribute before returning ----
        # Note: In the "files already exist" branch, self.item2tokens needs to be set
        if not hasattr(self, 'item2tokens'):
            self.item2tokens = item2tokens
        return item2tokens

    def _create_reverse_mapping(self):
        """Create reverse mapping for inference"""
        tokens2item = {}
        for item, tokens in self.item2tokens.items():
            item_id = self.dataset.item2id[item]
            tokens2item[tuple(tokens)] = item_id
        return tokens2item

    def _save_mappings(self):
        """Save mapping files"""
        # Build path - Fix: Use class name and category
        dataset_name = self.dataset.__class__.__name__  # Use class name, e.g., "AmazonReviews2014"

        # Include category in path if the category attribute exists
        if hasattr(self.dataset, 'category') and self.dataset.category:
            cache_dir = os.path.join(
                self.dataset.cache_dir, 'processed'
            )
        else:
            cache_dir = os.path.join(
                'data', dataset_name, 'processed'
            )

        os.makedirs(cache_dir, exist_ok=True)

        # 🚀 Filename includes: model+PCA+quantizer tag(+seed/iters)+n_digit, entirely avoiding configuration conflicts
        model_basename = os.path.basename(self.config["sent_emb_model"])
        quant_tag = self.index_factory
        if self.sid_quantizer == 'rq_kmeans':
            quant_tag += f'_seed{self.config.get("rq_kmeans_seed",1234)}_it{self.config.get("rq_kmeans_niters",20)}'
        elif self.sid_quantizer == 'none':
            quant_tag += f'_seed{self.config.get("sid_random_seed",12345)}'
        map_tag = f'{model_basename}_pca{self.config["sent_emb_pca"]}_{quant_tag}_{self.n_digit}d'

        # Save forward index: item_id → SID-tokens
        item_id2tokens = np.zeros((self.dataset.n_items, self.n_digit), dtype=np.int64)
        for item, tokens in self.item2tokens.items():
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = np.array(tokens)

        np.save(os.path.join(cache_dir, f'item_id2tokens_{map_tag}.npy'), item_id2tokens)

        # Save inverted index: SID-tokens → item_id
        with open(os.path.join(cache_dir, f'tokens2item_{map_tag}.pkl'), 'wb') as f:
            pickle.dump(self.tokens2item, f)

        self.log(f'[TOKENIZER] Saved mappings with tag: {map_tag} to {cache_dir}')
        self.log(f'[TOKENIZER] Files: item_id2tokens_{map_tag}.npy, tokens2item_{map_tag}.pkl')

    def encode_history(self, item_seq, max_len=None):
        """Encode user history sequence"""
        if max_len is None:
            max_len = self.config.get('max_history_len', 50)
        if len(item_seq) > max_len:
            item_seq = item_seq[-max_len:]

        history_sid = []
        for item in item_seq:
            if item in self.item2tokens:
                # Convert offset-adjusted token IDs back to codebook IDs (0..K-1)
                tokens = list(self.item2tokens[item])  # Offset-adjusted token IDs
                codebook_ids = []
                for digit, token_id in enumerate(tokens):
                    codebook_id = token_id - (self.sid_offset + digit * self.codebook_size)
                    codebook_ids.append(codebook_id)
                history_sid.append(codebook_ids)
            else:
                # Fill unknown items with PAD (use -1 as a sentinel value for PAD to avoid confusion with codebook_id=0)
                history_sid.append([-1] * self.n_digit)

        # Pad to fixed length
        while len(history_sid) < max_len:
            history_sid.append([-1] * self.n_digit)

        return history_sid  # Returns list, letting datasets.map handle tensor conversion automatically

    def encode_history_with_mask(self, item_seq, max_len=None):
        """Encode user history sequence and return padding mask simultaneously"""
        if max_len is None:
            max_len = self.config.get('max_history_len', 50)
        if len(item_seq) > max_len:
            item_seq = item_seq[-max_len:]

        history_sid = []
        history_mask = []  # True=valid position, False=PAD position

        for item in item_seq:
            if item in self.item2tokens:
                # Convert offset-adjusted token IDs back to codebook IDs (0..K-1)
                tokens = list(self.item2tokens[item])  # Offset-adjusted token IDs
                codebook_ids = []
                for digit, token_id in enumerate(tokens):
                    codebook_id = token_id - (self.sid_offset + digit * self.codebook_size)
                    codebook_ids.append(codebook_id)
                history_sid.append(codebook_ids)
                history_mask.append(True)  # Valid position
            else:
                # Fill unknown items with PAD (use -1 as a sentinel value for PAD to avoid confusion with codebook_id=0)
                history_sid.append([-1] * self.n_digit)
                history_mask.append(False)  # PAD position

        # Pad to fixed length
        while len(history_sid) < max_len:
            history_sid.append([-1] * self.n_digit)
            history_mask.append(False)  # PAD position

        return history_sid, history_mask  # Returns list, letting datasets.map handle tensor conversion automatically

    def encode_decoder_input(self, target_item):
        """Encode decoder input - remains aligned with RPG_ED"""
        if target_item in self.item2tokens:
            tokens = list(self.item2tokens[target_item])  # 4 token IDs (with offset)

            # Convert token IDs to codebook IDs
            codebook_tokens = []
            for digit, token_id in enumerate(tokens):
                codebook_id = token_id - (self.sid_offset + digit * self.codebook_size)
                codebook_tokens.append(codebook_id)

            # Both decoder input and labels are codebook IDs
            decoder_input = codebook_tokens  # [cb0, cb1, cb2, cb3]
            decoder_labels = codebook_tokens  # [cb0, cb1, cb2, cb3]
        else:
            # Unknown item
            decoder_input = [self.pad_token] * self.n_digit  # Length n_digit
            decoder_labels = [self.pad_token] * self.n_digit  # Length n_digit

        return decoder_input, decoder_labels

    def decode_tokens_to_item(self, tokens):
        """Decode a sequence of tokens into an item ID"""
        if len(tokens) != self.n_digit:
            return None

        token_tuple = tuple(tokens)
        return self.tokens2item.get(token_tuple)

    def codebooks_to_item_id(self, cb_ids):
        """
        Convert a sequence of codebook IDs to an item_id, validating correctness

        Args:
            cb_ids: List[int] of length n_digit, original codebook IDs (0-255)

        Returns:
            item_id (int) or None (if invalid)
        """
        if len(cb_ids) != self.n_digit:
            return None

        # Convert codebook IDs to token IDs
        token_ids = [
            cb_ids[d] + self.sid_offset + d * self.codebook_size
            for d in range(self.n_digit)
        ]

        # Look up corresponding item_id
        return self.tokens2item.get(tuple(token_ids))

    def tokenize_function(self, example: dict, split: str) -> dict:
        """Tokenize function - fixes data leakage issues"""
        item_seq = example['item_seq']  # Python list
        target_item = item_seq[-1]  # Original string

        # Fix: All splits should use item_seq[:-1] as history to prevent data leakage
        history_sid, history_mask = self.encode_history_with_mask(item_seq[:-1])

        if split == 'train':
            # Encode decoder inputs during training
            decoder_input, decoder_labels = self.encode_decoder_input(target_item)
            return {
                'history_sid': history_sid,  # Direct list
                'history_mask': history_mask,  # Direct list
                'decoder_input_ids': decoder_input,  # Direct list
                'decoder_labels': decoder_labels  # Direct list
            }
        else:
            # Generate ground truth labels during validation/testing
            _, decoder_labels = self.encode_decoder_input(target_item)
            return {
                'history_sid': history_sid,  # Direct list
                'history_mask': history_mask,  # Direct list
                'labels': decoder_labels  # New addition: True label sequence
            }

    def tokenize(self, datasets: dict) -> dict:
        """Tokenize the datasets"""
        tokenized_datasets = {}
        for split in datasets:
            tokenized_datasets[split] = datasets[split].map(
                lambda t: self.tokenize_function(t, split),
                batched=False,  # Turn off batching to prevent structural chaos in the data
                remove_columns=datasets[split].column_names,
                num_proc=self.config['num_proc'],
                desc=f'Tokenizing {split} set: '
            )

        for split in datasets:
            tokenized_datasets[split].set_format(type='torch')

        return tokenized_datasets

    # ====== New feature: SID→items mapping and utility tools ======
    def _sid_tokens_to_cb_tuple(self, tokens):
        """
        Convert offset-adjusted SID tokens (length n_digit) into a codebook index tuple (each slot 0..K-1).
        For example: [sid_offset + 0*K + a, sid_offset + 1*K + b, ...] → (a, b, ...)
        """
        assert len(tokens) == self.n_digit
        cb = []
        for d, tok in enumerate(tokens):
            cb.append(int(tok) - (self.sid_offset + d * self.codebook_size))
        return tuple(cb)

    def _build_cb2items_map(self):
        """
        Construct an inverted table mapping SID combinations → items based on self.item2tokens.
        Note: One-to-many mappings are allowed (if a "collision" occurs, include all elements).
        """
        from collections import defaultdict
        cb2items = defaultdict(list)
        for item, toks in self.item2tokens.items():
            cb = self._sid_tokens_to_cb_tuple(toks)
            cb2items[cb].append(item)
        return cb2items

    @property
    def cb2items(self):
        """
        Lazy caching: Constructs the SID→items mapping upon first access and caches it to _cb2items
        """
        if not hasattr(self, "_cb2items") or self._cb2items is None:
            self._cb2items = self._build_cb2items_map()
        return self._cb2items

    def cb_tuple_to_item_ids(self, cb):
        """
        Given a codebook tuple, return the corresponding item_id list (stable following building order).
        """
        items = self.cb2items.get(cb, [])
        out = []
        for it in items:
            iid = self.item2id.get(it, 0)
            if iid > 0:
                out.append(iid)
        return out
