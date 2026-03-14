import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ==========================================
# 1. 新核心：O(N) 复杂度的线性注意力机制
# ==========================================
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # 🌟 核心创新点：用 ELU+1 替代 Softmax，将 $O(N^2)$ 降维打击到 $O(N)$
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # 改变矩阵乘法顺序
        kv = torch.einsum('b h n d, b h n m -> b h d m', k, v)
        z = 1 / (torch.einsum('b h n d, b h d -> b h n', q, k.sum(dim=2)) + 1e-6)
        out = torch.einsum('b h n d, b h d m, b h n -> b h n m', q, kv, z)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """
    标准的 Pre-Norm Transformer 编码器块
    """

    def __init__(self, dim, heads, dim_head, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearAttention(dim, heads=heads, dim_head=dim_head)

        # 🌟 新增：用于残差连接前的 Dropout
        self.drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # 带有残差连接的 Pre-Norm 结构
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class LinearSpatiotemporalTransformer(nn.Module):
    def __init__(self, in_channels=1, patch_size=8, embed_dim=128, img_size=96, depth=3, out_channels=16, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )

        self.pos_embed = nn.Parameter(torch.randn(1, 16 * self.num_patches, embed_dim))  # 这个16是时间序列的长度

        # 🌟 新增：位置编码后的 Dropout
        self.pos_drop = nn.Dropout(p=dropout)

        # 🌟 修复点 1：使用刚写好的 TransformerBlock
        self.layers = nn.ModuleList([
            TransformerBlock_new(dim=embed_dim, heads=6, dim_head=64, dropout=dropout)
            for _ in range(depth)
        ])

        # self.to_hidden_map = nn.Sequential(
        #     nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1),
        #     nn.ReLU(), )

    def forward(self, S_t):
        x = S_t.transpose(1, 2)
        x = self.patch_embed(x)
        B, C, T, H_p, W_p = x.shape

        x = rearrange(x, 'b c t h w -> b (t h w) c')
        x = x + self.pos_embed[:, :x.shape[1], :]

        # 🌟 在进入 Transformer Block 之前应用 Dropout
        x = self.pos_drop(x)

        # 🌟 修复点 2：极其清爽的层级调用
        for block in self.layers:
            x = block(x)

        x = rearrange(x, 'b (t h w) c -> b t h w c', t=T, h=H_p, w=W_p)
        H_t = x[:, -1, :, :, :]
        H_t = H_t.permute(0, 3, 1, 2)

        H_t = F.interpolate(H_t, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        # H_t = self.to_hidden_map(H_t)

        return H_t


# ==========================================
# 新增：融入 TCN 灵魂的卷积前馈网络 (ConvFFN)
# ==========================================
class ConvFFN(nn.Module):
    """
    带有局部感受野的卷积前馈网络
    完美平替传统 Transformer 中的纯 Linear FFN
    """

    def __init__(self, dim, expansion_factor=4, kernel_size=3):
        super().__init__()
        inner_dim = dim * expansion_factor

        # 1. 升维
        self.fc1 = nn.Linear(dim, inner_dim)

        # 2. 深度可分离卷积 (Depthwise Conv1d)：专门捕捉相邻时间步的“局部突变”
        # padding=kernel_size//2 保证序列长度不变
        # groups=inner_dim 极大降低参数量，防止过拟合
        self.conv = nn.Conv1d(
            in_channels=inner_dim,
            out_channels=inner_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=inner_dim
        )

        # 3. 激活函数与降维
        self.act = nn.GELU()
        self.fc2 = nn.Linear(inner_dim, dim)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Dim)
        x = self.fc1(x)

        # 为了适应 Conv1d，必须把维度转成 (Batch, Channels, Seq_Len)
        x = rearrange(x, 'b n d -> b d n')

        # 提取局部时间/空间的平滑与突变特征
        x = self.conv(x)

        # 换回 Transformer 喜欢的形状 (Batch, Seq_Len, Dim)
        x = rearrange(x, 'b d n -> b n d')

        x = self.act(x)
        x = self.fc2(x)
        return x


# ==========================================
# 升级版：融合全局与局部的 ConvTransformerBlock
# ==========================================
class TransformerBlock_new(nn.Module):
    """
    全局注意力 + 局部卷积 的终极融合块
    (由于主程序调用的名字叫 TransformerBlock，我们直接复用这个名字，方便你无缝替换)
    """

    def __init__(self, dim, heads, dim_head, kernel_size=3, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        # 全局视野：Linear Attention 捕捉长程依赖
        self.attn = LinearAttention(dim, heads=heads, dim_head=dim_head)

        self.norm2 = nn.LayerNorm(dim)
        # 局部视野：ConvFFN 捕捉光伏/云层的瞬时突变
        self.ff = ConvFFN(dim=dim, kernel_size=kernel_size)

    def forward(self, x):
        # Stage 1: 全局信息交流
        x = x + self.drop(self.attn(self.norm1(x)))
        # Stage 2: 局部特征提炼
        x = x + self.drop(self.ff(self.norm2(x)))
        return x
