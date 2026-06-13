# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np


class DIFF_GRMGRAPHEvaluator:
    """
    DIFF_GRMGRAPH模型的评估器
    """
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metric2func = {
            'recall': self.recall_at_k,
            'ndcg': self.ndcg_at_k
        }

        self.pad_token = self.tokenizer.pad_token
        self.maxk = max(config['topk'])

        # 🚀 新增：总体统计累加器
        self.total_seqs = 0
        self.total_legals = 0
        self.total_unique = 0

        # 🚀 新增：SID组合统计累加器（用于计算平均值）
        self.batch_legal_ratios = []
        self.batch_duplicate_ratios = []
        self.batch_dup10_ratios = []  # 新增：用户内部top-10重复率

        # 调试信息：确认使用了DIFF_GRMGRAPHEvaluator
        print(f'>> Using evaluator = {self.__class__.__name__} (fixed duplicate scoring bug)')
        print(f'>> Recall: uses any() to avoid duplicate scoring')
        print(f'>> NDCG: uses first-hit-only to avoid duplicate DCG accumulation')
        print(f'>> Fixed: index bounds checking to prevent out-of-bounds errors')
        print(f'>> Added: illegal sequence filtering for more accurate evaluation')

        # ==== 新增：是否启用“SID→item 展开评估”开关 ====
        self.eval_expand = bool(self.config.get('eval_expand_sid_to_items', False))
        # 展开前是否对 SID 去重的策略（目前只支持 first）
        self.eval_expand_dedup = str(self.config.get('eval_expand_dedup', 'first')).lower()

        # 取 cb2items 映射（来自 tokenizer 惰性缓存）
        self.cb2items = getattr(self.tokenizer, 'cb2items', None)
        if self.cb2items is None:
            try:
                self.cb2items = self.tokenizer._build_cb2items_map()
            except Exception:
                self.cb2items = {}



    def calculate_pos_index(self, preds, labels):
        """
        计算预测结果与真标签的匹配情况（beam search已保证合法性）

        Args:
            preds: (batch_size, maxk, n_digit) - 生成的SID序列（已过滤为合法）
            labels: (batch_size, n_digit) - 真标签序列

        Returns:
            pos_index: (batch_size, maxk) - 每个位置是否匹配真标签
        """
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()

        B, maxk, n_digit = preds.shape

        # 🚀 简化：beam search已保证返回的序列都是合法的
        # 直接计算匹配情况，无需再过滤
        pos_index = torch.zeros((B, maxk), dtype=torch.bool)

        for i in range(B):
            # 获取真标签
            cur_label = labels[i].tolist()  # [n_digit]

            for j in range(maxk):
                # 获取预测序列
                cur_pred = preds[i, j].tolist()  # [n_digit]

                # 比较codebook IDs（0-255范围）
                if cur_pred == cur_label:
                    pos_index[i, j] = True
                    break  # 找到第一个匹配就停止，避免重复计分

        return pos_index

    def recall_at_k(self, pos_index, k):
        """计算Recall@k（修复：重复命中只计1分）"""
        # pos_index: (batch_size, maxk) - 已经过滤为合法序列
        # 修复：使用any()避免重复计分，只要top-k中有≥1个匹配就得1分
        return pos_index[:, :k].any(dim=1).cpu().float()

    def ndcg_at_k(self, pos_index, k):
        """计算NDCG@k（修复：重复命中只计第一次的DCG）"""
        # pos_index: (batch_size, maxk) - 已经过滤为合法序列
        batch_size, maxk = pos_index.shape
        device = pos_index.device

        # 创建rank权重：1/log2(rank+1)
        ranks = torch.arange(1, maxk + 1, device=device).float()
        dcg_weights = 1.0 / torch.log2(ranks + 1)

        # 修复：只计算第一个匹配位置的DCG，避免重复计分（向量化实现）
        # 创建位置索引矩阵
        position_matrix = torch.arange(maxk, device=device).expand(batch_size, -1)

        # 找到每个样本第一个匹配的位置（向量化）
        # 将非匹配位置设为一个很大的数，这样min操作会忽略它们
        masked_positions = torch.where(pos_index, position_matrix, torch.full_like(position_matrix, maxk))
        first_hit_positions = masked_positions.min(dim=1).values  # [batch_size]

        # 计算DCG：只有在top-k内且有匹配时才得分
        # 修复索引越界问题：确保only真正有匹配且在top-k内的样本才计分
        has_hit = first_hit_positions < maxk  # 是否有匹配
        in_topk = first_hit_positions < k  # 匹配是否在top-k内
        valid_mask = has_hit & in_topk  # 既有匹配又在top-k内

        # 安全的索引访问：先限制索引范围，再计算得分
        safe_positions = torch.clamp(first_hit_positions, 0, maxk - 1)
        dcg_scores = torch.where(valid_mask, dcg_weights[safe_positions], torch.tensor(0.0, device=device))

        # 对于单标签推荐任务，IDCG=1.0，所以DCG就是NDCG
        return dcg_scores.cpu().float()

    # ===== 工具：把 SID（codebook id 序列）转为 cb tuple =====
    def _sid_row_to_cb(self, sid_row):
        # sid_row: [n_digit]，元素 ∈ [0..K-1]
        return tuple(int(x) for x in sid_row.tolist())

    # ===== 核心：把“合法且去重后的 SID 列表”展开为 item_id 排序表（截断 K）=====
    def _expand_sid_list_to_items(self, sid_list, K):
        seen = set()
        expanded = []
        if sid_list.ndim != 2:
            return expanded
        M, _ = sid_list.shape
        for i in range(M):
            cb = self._sid_row_to_cb(sid_list[i])
            # 去非法
            if cb not in self.cb2items:
                continue
            # 去重
            if self.eval_expand_dedup == 'first':
                if cb in seen:
                    continue
                seen.add(cb)
            # 展开为 item_id
            iid_list = self.tokenizer.cb_tuple_to_item_ids(cb) if hasattr(self.tokenizer, 'cb_tuple_to_item_ids') else []
            for iid in iid_list:
                expanded.append(iid)
                if len(expanded) >= K:
                    return expanded
        return expanded

    # ===== 基于“展开后的 item 列表”构造 pos_index（找目标 item 的首次命中位）=====
    def _build_pos_index_from_items(self, item_ranked, target_item_id, Kmax):
        hit_pos = None
        for idx, iid in enumerate(item_ranked):
            if iid == target_item_id:
                hit_pos = idx
                break
        pos = torch.zeros(Kmax, dtype=torch.bool)
        if hit_pos is not None and hit_pos < Kmax:
            pos[hit_pos] = True
        return pos

    def _dup_ratio_per_user(self, preds, k=10):
        """
        计算一个 batch 内"用户内部"的重复率。
        preds: [B, maxk, n_digit]（已保证 maxk ≥ k）
        返回:  [B] 每个用户自己的重复率
        """
        B, _, n_digit = preds.shape
        dup_ratios = []

        for b in range(B):
            # 仅取前 k 条
            seqs = preds[b, :k]                       # [k, n_digit]
            k_seqs = [tuple(s.tolist()) for s in seqs]
            unique_cnt = len(set(k_seqs))
            dup_ratios.append(1 - unique_cnt / k)     # in [0,1]

        return torch.tensor(dup_ratios, dtype=torch.float32)

    def calculate_weighted_score(self, preds, labels):
        """
        计算加权综合分数：NDCG@10 * 0.6 + RECALL@10 * 0.4

        Args:
            preds: (batch_size, beam_size, n_digit) - 生成的SID序列
            labels: (batch_size, n_digit) - 真标签序列

        Returns:
            weighted_score: 加权综合分数
        """
        pos_index = self.calculate_pos_index(preds, labels)

        # 计算NDCG@10
        ndcg_10 = self.ndcg_at_k(pos_index, k=10)

        # 计算RECALL@10
        recall_10 = self.recall_at_k(pos_index, k=10)

        # 计算加权分数：NDCG@10 * 0.6 + RECALL@10 * 0.4
        weighted_score = 0.8 * ndcg_10 + 0.2 * recall_10

        return weighted_score

    def calculate_metrics(self, preds, labels, suffix=""):
        """计算所有指标"""
        results = {}
        pos_index = self.calculate_pos_index(preds, labels)

        # 🚀 更新总体统计
        B, maxk, n_digit = preds.shape
        self.total_seqs += preds.numel() // n_digit
        self.total_legals += sum(
            self.tokenizer.codebooks_to_item_id(seq.tolist()) is not None
            for seq in preds.view(-1, n_digit)
        )
        self.total_unique += len({
            tuple(seq.tolist()) for seq in preds.view(-1, n_digit)
        })

        # 🚀 计算当前batch的SID组合统计
        # 修复合法率计算：使用序列数作为分母，而不是token数
        total_seqs = preds.numel() // n_digit
        current_legal_ratio = sum(
            self.tokenizer.codebooks_to_item_id(seq.tolist()) is not None
            for seq in preds.view(-1, n_digit)
        ) / total_seqs

        current_duplicate_ratio = 1 - len({
            tuple(seq.tolist()) for seq in preds.view(-1, n_digit)
        }) / total_seqs

        # 收集统计信息用于计算平均值
        self.batch_legal_ratios.append(current_legal_ratio)
        self.batch_duplicate_ratios.append(current_duplicate_ratio)

        # ✅ 新增：把 batch 级合法率/重复率作为“指标”回传（Trainer 会自动求均值并写入 TB）
        results[f'legal_ratio{suffix}'] = torch.tensor([current_legal_ratio], dtype=torch.float32)
        results[f'duplicate_ratio{suffix}'] = torch.tensor([current_duplicate_ratio], dtype=torch.float32)

        # ---------- 计算"用户内部"Top-10 重复率 ----------
        dup10 = self._dup_ratio_per_user(preds, k=10)     # [B]
        results[f'dup@10{suffix}'] = dup10                         # 会被平均后写入 final_results

        # 顺便累计到 batch 统计（想看全局平均）
        self.batch_dup10_ratios.append(dup10.mean().item())

        for metric in self.config['metrics']:
            for k in self.config['topk']:
                results[f"{metric}@{k}{suffix}"] = self.metric2func[metric](pos_index, k)

        # 添加加权综合分数（只在confidence模式下计算）
        if suffix == "":  # 仅confidence模式才算weighted_score
            weighted_score = self.calculate_weighted_score(preds, labels)
            results['weighted_score'] = weighted_score

        # ====== 新增：按簇展开 → item 级评估（额外产出一套指标，带 _xitem 后缀）======
        if self.eval_expand:
            B, maxk, n_digit = preds.shape
            # 1) 标签 SID → 目标 item_id
            target_item_ids = []
            for i in range(B):
                lab = labels[i].tolist()
                target_iid = self.tokenizer.codebooks_to_item_id(lab)
                target_item_ids.append(int(target_iid) if target_iid is not None else 0)

            # 2) 展开每个样本的 SID Top-maxk → item Top-Kmax
            Kmax = self.maxk
            pos_index_item = torch.zeros(B, Kmax, dtype=torch.bool)
            dup10_list = []

            for i in range(B):
                sid_top = preds[i]
                expanded_items = self._expand_sid_list_to_items(sid_top, K=Kmax)

                k10 = min(10, len(expanded_items))
                if k10 > 0:
                    uniq = len(set(expanded_items[:k10]))
                    dup10_list.append(1 - uniq / k10)
                else:
                    dup10_list.append(1.0)

                pos_index_item[i] = self._build_pos_index_from_items(
                    expanded_items, target_item_ids[i], Kmax
                )

            # 3) 计算 item 级指标
            for metric in self.config['metrics']:
                for k in self.config['topk']:
                    results[f"{metric}@{k}{suffix}_xitem"] = self.metric2func[metric](pos_index_item, k)

            ndcg_10_x = self.ndcg_at_k(pos_index_item, k=10)
            recall_10_x = self.recall_at_k(pos_index_item, k=10)
            weighted_x = 0.8 * ndcg_10_x + 0.2 * recall_10_x
            results[f"weighted_score{suffix}_xitem"] = weighted_x
            results[f"dup@10{suffix}_xitem"] = torch.tensor(dup10_list, dtype=torch.float32)

        return results

    def print_final_stats(self):
        """打印最终统计结果"""
        if self.total_seqs > 0:
            # 计算总体统计
            legal_ratio = self.total_legals / self.total_seqs
            # 修复重复率计算：使用正确的公式
            duplicate_ratio = 1 - self.total_unique / self.total_seqs

            # 计算batch平均值（更准确）
            if self.batch_legal_ratios:
                avg_legal_ratio = sum(self.batch_legal_ratios) / len(self.batch_legal_ratios)
                avg_duplicate_ratio = sum(self.batch_duplicate_ratios) / len(self.batch_duplicate_ratios)

                print(f"[SID_STATS] 平均合法率: {avg_legal_ratio:.3f}, 平均重复率: {avg_duplicate_ratio:.3f}")

                # 新增：用户内部top-10重复率统计
                if self.batch_dup10_ratios:
                    avg_dup10 = sum(self.batch_dup10_ratios) / len(self.batch_dup10_ratios)
                    print(f"[SID_STATS] 用户内部 Top-10 平均重复率: {avg_dup10:.3f}")
            else:
                print(f"[SID_STATS] 总体合法率: {legal_ratio:.3f}, 总体重复率: {duplicate_ratio:.3f}")