import torch
import torch.nn as nn

from model.transformer import TransformerBlock


# ==========================================
# 2. 时序支路：Linear Time-Series Transformer
# ==========================================
class TimeSeriesBranch(nn.Module):
    """
    用 Linear Transformer 完美平替 GRU
    处理历史数值数据 (如: Active Power, Clear-Sky GHI, Solar Zenith)
    """

    def __init__(self, input_dim=3, seq_len=16, embed_dim=384, depth=3, heads=6, dim_head=64, final_dim=128,
                 dropout=0.1):
        super().__init__()

        # 1. 独立特征映射 (Embedding): 将 3 维特征升维到 Transformer 喜欢的高维空间
        self.embed = nn.Linear(input_dim, embed_dim)

        # 2. 位置编码 (Positional Encoding): 告诉 Transformer 序列的先后顺序
        # GRU 天生有顺序，但 Transformer 是“全连接”的，必须加位置编码防遗忘
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim))

        # 🌟 新增：位置编码后的 Dropout
        self.pos_drop = nn.Dropout(p=dropout)

        # 3. 线性 Transformer 层堆叠
        self.layers = nn.ModuleList([
            TransformerBlock(dim=embed_dim, heads=heads, dim_head=dim_head, dropout=dropout)
            for _ in range(depth)
        ])

        # 4. 输出映射 (Linear head)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, final_dim)
        )

    def forward(self, x):
        """
        x shape: (Batch, Seq_Len, Input_Dim) -> e.g., (Batch, 16, 3)
        """
        # A. 嵌入与位置编码
        x = self.embed(x)  # -> (Batch, 16, embed_dim)
        x = x + self.pos_embed  # 加上时间顺序信息

        # 🌟 在进入 Transformer Block 之前应用 Dropout
        x = self.pos_drop(x)

        # B. Transformer 全局时序特征提取 (并行计算，比 GRU 快得多！)
        for block in self.layers:
            x = block(x)  # -> (Batch, 16, embed_dim)

        # C. 提取序列特征
        # (完全对齐你之前 GRU 取最后一个时间步的操作，因为 Attention 已经让最后一步包含了前面所有的上下文)
        last_step_out = x[:, -1, :]  # -> (Batch, embed_dim)

        # D. 输出最终特征
        ts_feature = self.head(last_step_out)  # -> (Batch, final_dim)

        return ts_feature


# ==========================================
# 测试代码 (模拟输入并验证形状)
# ==========================================
if __name__ == "__main__":
    print("🚀 开始测试全新 Linear TimeSeries Transformer...")

    batch_size = 2
    seq_len = 16
    input_dim = 3
    final_dim = 64

    model = TimeSeriesBranch(
        input_dim=input_dim,
        seq_len=seq_len,
        embed_dim=64,  # 内部特征升维
        depth=2,  # 堆叠 2 层 Transformer
        final_dim=final_dim
    )

    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    print(f"\n📥 输入序列形状 : {dummy_input.shape}")

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\n📤 最终输出特征形状 : {output.shape}")

    if output.shape == (batch_size, final_dim):
        print("\n✅ 测试完美通过！GRU 已被彻底淘汰。")
        print("💡 模型已成功利用全局自注意力机制 (Self-Attention)，")
        print("   一次性并行看穿 16 个时刻的数值动态变化。")
    else:
        print("\n❌ 测试失败，输出形状不符合预期。")
