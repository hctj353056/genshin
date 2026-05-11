# hex_mha_module_v2.py
# 优化版 16进制多头自注意力模块
# 特点：位置编码、层归一化、非线性激活、因果掩码、KV-Cache、训练支持

import numpy as np
from typing import Optional, Tuple

HEX_CHARS = '0123456789ABCDEF'
HEX_TO_IDX = {c: i for i, c in enumerate(HEX_CHARS)}
IDX_TO_HEX = {i: c for i, c in enumerate(HEX_CHARS)}


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / (e.sum(axis=axis, keepdims=True) + 1e-9)


def cross_entropy_loss(logits: np.ndarray, target_indices: np.ndarray) -> float:
    """计算交叉熵损失，logits形状 (seq_len, vocab_size)，target_indices形状 (seq_len,)"""
    probs = softmax(logits, axis=-1)
    seq_len = logits.shape[0]
    loss = 0.0
    for i in range(seq_len):
        loss -= np.log(probs[i, target_indices[i]] + 1e-9)
    return loss / seq_len


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


class HexMHA:
    def __init__(self, seq_len: int = 8, dim: int = 64, heads: int = 4, embed_dim: int = 64,
                 dropout: float = 0.0, seed: int = 42, mode: str = 'streaming', causal: bool = False):
        """
        seq_len: 最大序列长度（用于权重初始化位置编码）
        dim: 注意力维度（Q、K、V 总维度）
        heads: 注意力头数（dim 必须被 heads 整除）
        embed_dim: 嵌入维度（输入输出维度）
        dropout: 暂未实现（保留接口）
        seed: 随机种子
        mode: 'streaming' 或 'cache'
        causal: 是否使用因果掩码（缓存模式下自动启用）
        """
        assert dim % heads == 0
        self.seq_len = seq_len
        self.dim = dim
        self.heads = heads
        self.d_k = dim // heads
        self.embed_dim = embed_dim
        self.mode = mode
        self.causal = causal

        rng = np.random.default_rng(seed)

        # 可学习的字符嵌入 (16, embed_dim)
        # 初始化为独热码，便于学习"复制"
        self.token_embed = np.eye(16, embed_dim, dtype=np.float32)

        # 可学习的位置嵌入 (seq_len, embed_dim)
        self.pos_embed = rng.normal(scale=0.1, size=(seq_len, embed_dim)).astype(np.float32)

        # QKV 投影 (embed_dim -> dim)
        self.Wq = rng.normal(scale=0.1, size=(embed_dim, dim)).astype(np.float32)
        self.Wk = rng.normal(scale=0.1, size=(embed_dim, dim)).astype(np.float32)
        self.Wv = rng.normal(scale=0.1, size=(embed_dim, dim)).astype(np.float32)

        # 输出投影 (dim -> embed_dim)
        self.Wo = rng.normal(scale=0.1, size=(dim, embed_dim)).astype(np.float32)

        # LayerNorm 参数 (两个：注意力输出后和最终输出前)
        self.ln1_gamma = np.ones(embed_dim, dtype=np.float32)
        self.ln1_beta = np.zeros(embed_dim, dtype=np.float32)
        self.ln2_gamma = np.ones(embed_dim, dtype=np.float32)
        self.ln2_beta = np.zeros(embed_dim, dtype=np.float32)

        # 最终分类头：embed_dim -> 16
        # 初始化为token_embed的转置，便于实现"复制"
        self.classifier = self.token_embed.T.copy()

        # 缓存状态（用于增量推理）
        self.cache_k: Optional[np.ndarray] = None
        self.cache_v: Optional[np.ndarray] = None
        self.cache_seq_len = 0

    def _embed(self, hex_str: str) -> np.ndarray:
        """将 hex 字符串转为嵌入向量 (L, embed_dim)，L = len(hex_str)"""
        indices = [HEX_TO_IDX[c.upper()] for c in hex_str if c.upper() in HEX_TO_IDX]
        if not indices:
            raise ValueError("输入字符串不包含合法的十六进制字符")
        x = self.token_embed[indices]  # (L, embed_dim)

        # 位置编码（支持长度小于等于预先设置的最大长度）
        L = x.shape[0]
        if L > self.pos_embed.shape[0]:  # 超出预置长度时，用零填充或截断，此处简单截断
            L = self.pos_embed.shape[0]
            x = x[:L]
        x = x + self.pos_embed[:L]

        return x.astype(np.float32)

    def _forward_attention(self, x: np.ndarray, mask: Optional[np.ndarray] = None,
                           cache: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        核心注意力计算

        参数：
            x: (L, embed_dim)
            mask: 可选注意力掩码 (L, L) 或 (1, 1, L, Lk)
            cache: 是否使用 KV 缓存进行增量计算

        返回：
            out: (L, embed_dim) 注意力输出
            k: 更新后的 key 缓存 (用于增量计算)
            v: 更新后的 value 缓存
        """
        L = x.shape[0]  # 增量计算时，新输入长度可能为1

        if cache and self.cache_k is not None:
            # 只对新 token 计算 query
            q = x @ self.Wq  # (1, dim) 或 (L_new, dim)
            k_new = x @ self.Wk
            v_new = x @ self.Wv

            # 拼接历史 key/value（此时都是 dim 维度，未多头reshape）
            k_full = np.concatenate([self.cache_k, k_new], axis=0)  # (L_total, dim)
            v_full = np.concatenate([self.cache_v, v_new], axis=0)
            # 当前query的k/v
            k_q = k_new
            v_q = v_new
        else:
            # 全量计算
            q = x @ self.Wq  # (L, dim)
            k = x @ self.Wk
            v = x @ self.Wv
            k_full = k
            v_full = v
            k_q = k
            v_q = v

        # 多头重塑
        def reshape_multihead(tensor, head_dim):
            # tensor (seq, dim) -> (heads, seq, d_k)
            seq_len = tensor.shape[0]
            return tensor.reshape(seq_len, self.heads, self.d_k).transpose(1, 0, 2)

        q = reshape_multihead(q, self.d_k)  # (heads, Lq, d_k)
        
        # k/v 需要特殊处理：query用新token的k/v，key/value用完整的k/v
        k = reshape_multihead(k_full, self.d_k)  # (heads, L_total, d_k)
        v = reshape_multihead(v_full, self.d_k)  # (heads, L_total, d_k)
        k_new_mh = reshape_multihead(k_q, self.d_k)  # (heads, L_new, d_k)
        v_new_mh = reshape_multihead(v_q, self.d_k)  # (heads, L_new, d_k)

        # 缩放点积
        scores = q @ k.transpose(0, 2, 1) * (self.d_k ** -0.5)  # (heads, Lq, Lk)

        # 应用掩码
        if mask is not None:
            scores = scores + mask  # mask 中 -1e9 等，支持正确广播

        # softmax
        attn = softmax(scores, axis=-1)  # (heads, Lq, Lk)

        out = attn @ v  # (heads, Lq, d_k)

        # 合并多头
        out = out.transpose(1, 0, 2).reshape(L, self.dim)  # (L, dim)

        # 输出投影
        out = out @ self.Wo  # (L, embed_dim)

        # 更新缓存（存储多头reshape前的原始k/v）
        if cache:
            if self.cache_k is None:
                self.cache_k = k_q
                self.cache_v = v_q
            else:
                self.cache_k = np.concatenate([self.cache_k, k_q], axis=0)
                self.cache_v = np.concatenate([self.cache_v, v_q], axis=0)
            self.cache_seq_len += L

        return out, k, v

    def forward(self, hex_input: str, reset_cache: bool = False) -> str:
        """
        统一前向接口，根据 self.mode 调用相应模式
        """
        if self.mode == 'streaming':
            return self._streaming_forward(hex_input)
        elif self.mode == 'cache':
            return self._cache_forward(hex_input, reset=reset_cache)
        else:
            raise ValueError("不支持的模式")

    def _streaming_forward(self, hex_input: str) -> str:
        """流式模式：一次性处理整个输入"""
        x = self._embed(hex_input)  # (L, embed_dim)
        L = x.shape[0]

        # 构建因果掩码（如果需要）
        mask = None
        if self.causal:
            # 下三角矩阵，0 表示可见，负无穷表示屏蔽
            base_mask = np.tril(np.ones((L, L), dtype=np.float32))
            base_mask = (1.0 - base_mask) * -1e9
            # 3D mask: (heads, L, L)
            mask = np.broadcast_to(base_mask[np.newaxis, :, :], (self.heads, L, L))

        # 注意力
        attn_out, _, _ = self._forward_attention(x, mask=mask, cache=False)

        # 残差连接 + 层归一化
        x = layer_norm(x + attn_out, self.ln1_gamma, self.ln1_beta)

        # 前馈网络：线性 + ReLU
        ffn_out = np.maximum(0, x)  # 简单 ReLU，可扩展为两层

        # 最终层归一化
        x = layer_norm(ffn_out, self.ln2_gamma, self.ln2_beta)

        # 分类头
        logits = x @ self.classifier  # (L, 16)

        return self._decode_logits(logits)

    def _cache_forward(self, hex_input: str, reset: bool = False) -> str:
        """缓存模式：一次输入一个或多个 token，内部维护 KV 缓存，支持增量推理。

        返回当前已生成的所有 token 对应的 hex 字符串（基于最后一层 logits 解码）。
        """
        if reset:
            self.reset_cache()

        x = self._embed(hex_input)  # (L, embed_dim)
        L = x.shape[0]

        # 构建因果掩码（仅对新 token，禁止看到未来）
        mask = None
        if self.causal:
            total_len = self.cache_seq_len + L
            # 因果掩码: (L, total_len)，下三角为0可见，上三角为-inf屏蔽
            base_mask = np.tril(np.ones((L, total_len), dtype=np.float32))
            base_mask = (1.0 - base_mask) * -1e9  # 下三角=0，上三角=-inf
            # 3D mask: (heads, Lq, Lk)
            mask = np.broadcast_to(base_mask[np.newaxis, :, :], (self.heads, L, total_len))

        attn_out, _, _ = self._forward_attention(x, mask=mask, cache=True)

        # 残差 + LN（但增量时只有新 token，历史 token 已经处理过，故只对新 token 应用）
        new_x = layer_norm(x + attn_out, self.ln1_gamma, self.ln1_beta)
        ffn_out = np.maximum(0, new_x)
        final = layer_norm(ffn_out, self.ln2_gamma, self.ln2_beta)

        logits = final @ self.classifier  # (L, 16)

        # 解码并返回累积结果（此处仅解码新 token，为完整输出可保留历史解码）
        return self._decode_logits(logits)

    def _decode_logits(self, logits: np.ndarray) -> str:
        """将 logits 转换为 hex 字符串（argmax）"""
        indices = np.argmax(logits, axis=-1)
        return ''.join(IDX_TO_HEX[i] for i in indices)

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None
        self.cache_seq_len = 0

    def set_mode(self, mode: str):
        if mode not in ('streaming', 'cache'):
            raise ValueError("mode 必须是 'streaming' 或 'cache'")
        self.mode = mode
        self.reset_cache()

    def __repr__(self):
        return (f"HexMHA(seq_len={self.seq_len}, dim={self.dim}, heads={self.heads}, "
                f"embed_dim={self.embed_dim}, mode={self.mode}, causal={self.causal})")


# ============ 简单训练示例 ============
if __name__ == "__main__":
    np.random.seed(42)
    model = HexMHA(seq_len=16, dim=64, heads=4, embed_dim=64, mode='streaming', causal=True)

    # 生成随机训练数据：输入 -> 输出（将输入循环右移一位作为任务）
    def generate_batch(batch_size=4):
        inputs = []
        targets = []
        for _ in range(batch_size):
            length = np.random.randint(4, 8)
            in_seq = ''.join(np.random.choice(list(HEX_CHARS), size=length))
            # 目标：将整个序列循环右移一位
            out_seq = in_seq[-1] + in_seq[:-1]
            inputs.append(in_seq)
            targets.append(out_seq)
        return inputs, targets

    # 训练参数
    epochs = 100
    lr = 0.01

    print("开始训练（演示用随机任务，仅验证流程）...")
    for epoch in range(epochs):
        inputs, targets = generate_batch(batch_size=4)
        total_loss = 0.0
        grads_acc = {}  # 简单计数器演示，实际需用自动微分框架

        for inp, tgt in zip(inputs, targets):
            # 前向传播（流式）
            x = model._embed(inp)
            L = x.shape[0]
            # mask为3D: (heads, L, L)
            base_mask = np.tril(np.ones((L, L), dtype=np.float32))
            base_mask = (1.0 - base_mask) * -1e9
            mask = np.broadcast_to(base_mask[np.newaxis, :, :], (model.heads, L, L))
            attn_out, _, _ = model._forward_attention(x, mask=mask)
            x1 = layer_norm(x + attn_out, model.ln1_gamma, model.ln1_beta)
            ffn = np.maximum(0, x1)
            x2 = layer_norm(ffn, model.ln2_gamma, model.ln2_beta)
            logits = x2 @ model.classifier  # (L,16)

            # 计算损失（目标索引）
            tgt_indices = [HEX_TO_IDX[c] for c in tgt]
            loss = cross_entropy_loss(logits, np.array(tgt_indices))
            total_loss += loss
            # 反向传播（此处省略，建议迁移到 PyTorch 等框架）

        # 仅为了演示，不实际更新参数，所以 loss 不会下降
        # 打印损失
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: loss = {total_loss/4:.4f}")

    print("\n测试模型效果（随机权重，输出随机）:")
    test_in = "A6F0"
    out = model.forward(test_in)
    print(f"输入: {test_in} -> 输出: {out}")

    model.set_mode('cache')
    model.forward("DEAD", reset_cache=True)
    print("缓存模式逐步输入: 'DEAD' ->", model.forward("BE", reset_cache=False))
    print("缓存模式逐步输入: 'BE' ->", model.forward("EF", reset_cache=False))
