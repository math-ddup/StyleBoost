# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
# and https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py

import math

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


# FFN，前馈神经网络
def FeedForward(dim, mult=4):  # mult：特征维度扩展的倍数
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),  # 将输入维度从 dim 投影到更高维度 inner_dim
        nn.GELU(),  # 引入非线性
        nn.Linear(inner_dim, dim, bias=False),
    )
    # 输入的特征维度扩展为 dim * mult，通过非线性激活函数后，再将其映射回原始维度 dim。


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)？？ 应该是(bs, heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x
    # 刚开始张量是连续的，用view更高效，transpose操作后，可能变得不连续，用reshape更灵活且安全


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):  #  * 表示在它后面的参数（dim, dim_head, heads）必须以关键字形式指定。
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        # 将 x 和 latents 沿着倒数第二个维度（序列长度维度）进行拼接
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        # 将 kv_input 通过 to_kv 线性层映射为键（K）和值（V）。通过 chunk 操作将输出拆分为两个张量：一个用于键（K），一个用于值（V）。

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)
        # (batch_size, length, dim)-->(batch_size, heads, length, dim_per_head)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))  # sqrt是开根号
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        # weight.float()：将 weight 转换为 float32 类型，以避免精度问题；type(weight.dtype)：将结果恢复为原始的数据类型
        out = weight @ v  # @：矩阵乘法
        # (batch_size, heads, length, dim_per_head)

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)
        # (batch_size, heads, length, dim_per_head) --> (batch_size, length, heads, dim_per_head) --> (batch_size, length, heads*dim_per_head)

        return self.to_out(out)

# 主要来源于论文，设计用于处理序列数据，并通过注意力和前馈网络层进行特征转换。
class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        max_seq_len: int = 257,  # CLIP tokens + CLS token
        apply_pos_emb: bool = False,  # 位置嵌入
        num_latents_mean_pooled: int = 0,  # number of latents derived from mean pooled representation of the sequence 从平均池化序列生成的嵌入数量
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim) if apply_pos_emb else None
        # embedding大小为max_seq_len * embedding_dim

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        # 定义了可学习的查询向量，形状为 [1, num_queries, dim]。使用高斯分布初始化，并按 dim**0.5 进行缩放

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            # 定义一个模块  Rearrange：对张量的维度进行重新变换排序，“”里的是张量维度的映射关系
            if num_latents_mean_pooled > 0
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )
            # 创建一个列表，重复depth次

    def forward(self, x):
        if self.pos_emb is not None:
            n, device = x.shape[1], x.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            # 生成一个长度为 n 的整数序列 [0, 1, ..., n-1]然后将位置索引映射为位置嵌入
            x = x + pos_emb
            # 将位置编码加到原始输入 x 上，使序列数据带有位置信息

        latents = self.latents.repeat(x.size(0), 1, 1)
        # .size() 返回数组中的元素总数，0维复制x.size(0)次，其余保持不变

        x = self.proj_in(x)

        if self.to_latents_from_mean_pooled_seq:
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            # [:2]：获取x 的前两个维度，作为mask，然后计算x在指定维度1上的均值
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)
            # 沿着倒数第二个维度进行拼接

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            # 计算 x 和当前 latent 表示之间的注意力再加上latent
            latents = ff(latents) + latents
            # ff(latents)：对更新后的 latent 进行非线性变换

        latents = self.proj_out(latents)
        return self.norm_out(latents)

# 计算有掩码的加权均值，解决缺失数据或不规则序列的处理需求。
def masked_mean(t, *, dim, mask=None):
    if mask is None:
        return t.mean(dim=dim)
    # 如果没有提供 mask，直接计算均值。

    denom = mask.sum(dim=dim, keepdim=True)
    # mask 在指定维度 dim 上的元素求和，并通过 keepdim=True 保持该维度大小。防止除以0
    mask = rearrange(mask, "b n -> b n 1")
    # 重新排列mask维度：b n -> b n 1
    masked_t = t.masked_fill(~mask, 0.0)
    # 取反后的mask（即~mask），将原张量中~mask为True的位置填充为0.0

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)
    # clamp操作将denom中的每个元素限制在一个指定的范围内，这里是将所有元素的值至少设置为1e-5
