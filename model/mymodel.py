import torch
import torch.nn as nn

from model.visual_branch import VisualBranch
from model.time_series import TimeSeriesBranch


class MultiModalPVNet(nn.Module):
    def __init__(self, visual_dim=128, ts_dim=64, output_seq_len=4):
        super(MultiModalPVNet, self).__init__()

        # 实例化升级后的视觉支路
        self.visual_branch = VisualBranch(final_dim=visual_dim)

        self.ts_branch = TimeSeriesBranch(final_dim=ts_dim)

        fusion_dim = visual_dim + ts_dim
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_seq_len)
        )

    def forward(self, x_images, x_numeric):
        # x_images: (Batch, 16, 1, 96, 96)
        # x_numeric: (Batch, 16, 3)
        v_feat = self.visual_branch(x_images)

        # 适配我模拟的占位分支
        t_feat = self.ts_branch(x_numeric)

        combined_feat = torch.cat([v_feat, t_feat], dim=1)
        predictions = self.predictor(combined_feat)
        return predictions, v_feat, t_feat


# 测试块
if __name__ == "__main__":
    print("🚀 开始测试全新 Linear Transformer 多模态网络...")
    batch_size = 2
    seq_len = 16

    # 模拟数据
    dummy_imgs = torch.randn(batch_size, seq_len, 1, 96, 96)
    dummy_nums = torch.randn(batch_size, seq_len, 3)

    model = MultiModalPVNet(output_seq_len=4)
    model.eval()

    with torch.no_grad():
        output, _, _ = model(dummy_imgs, dummy_nums)

    print(f"\n📥 输入云图 : {dummy_imgs.shape}")
    print(f"📥 输入数值 : {dummy_nums.shape}")
    print(f"📤 最终预测 : {output.shape} (预期为 Batch={batch_size}, 预测步数=4)")

    if output.shape == (batch_size, 4):
        print("\n✅ Transformer 手术非常成功！没有任何排异反应！")
