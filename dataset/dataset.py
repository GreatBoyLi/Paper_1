import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from utils.config import load_config


class SatellitePVDataset(Dataset):
    count = 0
    def __init__(self, csv_path, satellite_dir,
                 input_seq_len=16, output_seq_len=4,
                 mode='train', split_ratio=0.8):
        """
        Args:
            csv_path: 包含24小时全天候连续数据的 CSV 路径
            satellite_dir: .npy 文件所在的根目录
            input_seq_len: 输入序列长度 (4小时 = 16个点)
            output_seq_len: 预测序列长度 (1小时 = 4个点)
            mode: 'train', 'val', or 'test'
        """
        self.input_len = input_seq_len
        self.output_len = output_seq_len
        self.satellite_dir = satellite_dir

        # 1. 读取 CSV (必须是包含黑夜的连续时间序列)
        self.df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
        self.df = self.df.sort_index()

        # 2. 划分数据集
        n = len(self.df)
        train_end = int(n * split_ratio)
        val_end = int(n * (split_ratio + 0.2))

        if mode == 'train':
            self.data = self.df.iloc[:train_end]
        elif mode == 'val':
            self.data = self.df.iloc[train_end:val_end]
        else:  # test
            self.data = self.df.iloc[val_end:]

        # ==========================================
        # 🌟 核心修改 1：严格的时间连续性校验
        # 允许跨夜，但绝对不允许中间缺失数据 (比如设备断电少了一天)
        # ==========================================
        self.valid_indices = []
        total_len = self.input_len + self.output_len
        # 15分钟分辨率下，N个点的时间跨度应该是 (N-1)*15 分钟
        expected_time_delta = pd.Timedelta(minutes=15 * (total_len - 1))

        max_possible_idx = len(self.data) - total_len
        for i in range(max_possible_idx + 1):
            start_time = self.data.index[i]
            end_time = self.data.index[i + total_len - 1]

            # 只有严格连续的时间段，才被认为是有效样本
            if end_time - start_time == expected_time_delta:
                self.valid_indices.append(i)

        print(f"[{mode}] 数据集加载完成 | 原始行数: {len(self.data)} | 严格连续的有效样本数: {len(self.valid_indices)}")

    def __len__(self):
        # 返回有效样本的数量
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # ==========================================
        # 🌟 获取校验过安全的真实行号
        # ==========================================
        real_idx = self.valid_indices[idx]

        # 1. 切片索引
        hist_start = real_idx
        hist_end = real_idx + self.input_len
        future_end = hist_end + self.output_len

        # 2. 获取数值特征 (Power, Zenith, GHI)
        features = ['Power_Norm', 'Clear_Sky_GHI', 'Solar_Zenith']
        x_numeric = self.data.iloc[hist_start:hist_end][features].values

        # 3. 获取预测目标 (未来功率)
        y_power = self.data.iloc[hist_end:future_end]['Power_Norm'].values

        # 🌟 核心修改 2：提取预测时间段的太阳天顶角，用于给 Loss 戴面具
        y_zenith = self.data.iloc[hist_end:future_end]['Solar_Zenith'].values

        # 4. 获取图像数据
        hist_timestamps = self.data.index[hist_start:hist_end]
        images = []
        for ts in hist_timestamps:
            file_name = f"sat_15min_{ts.strftime('%Y%m%d_%H%M')}.npy"

            yyyy = ts.strftime("%Y")
            mm = ts.strftime("%m")
            dd = ts.strftime("%d")
            yyyymm = f"{yyyy}{mm}"
            file_path = os.path.join(self.satellite_dir, yyyymm, dd, file_name)

            if os.path.exists(file_path):
                # 读取并归一化
                img = np.load(file_path).astype(np.float32)
                # ==========================================
                # 🛡️ 免疫系统开启：排查并消灭 NaN 和 Inf
                # ==========================================
                if np.isnan(img).any() or np.isinf(img).any():
                    # 策略 A：用整张图的有效平均值来填补坏掉的像素（最科学的物理做法）
                    valid_mean = np.nanmean(img[~np.isinf(img)])

                    # 如果这整张图彻彻底底全坏了（比如全屏 NaN）
                    if np.isnan(valid_mean):
                        valid_mean = 175.0  # 用背景物理下限兜底
                    # 将所有的 NaN 和 Inf 替换为这个安全的平均值
                    img = np.nan_to_num(img, nan=valid_mean, posinf=valid_mean, neginf=valid_mean)
                # ==========================================
                img = (img - 175.0) / (340.0 - 175.0)
                # 极限防爆：防止极个别数值归一化后越界
                img = np.clip(img, 0.0, 1.0)
                img = np.expand_dims(img, axis=0)
            else:
                # 🌟 全天候模式：找不到图大概率是晚上不拍了，直接全黑填充！
                img = np.zeros((1, 96, 96), dtype=np.float32)

            images.append(img)

        # 堆叠 -> (Seq, C, H, W)
        x_images = np.stack(images, axis=0)

        # 🌟 核心修改 3：在返回的字典中加入 y_zenith
        return {
            'x_images': torch.from_numpy(x_images).float(),  # Shape: (16, 1, 96, 96)
            'x_numeric': torch.from_numpy(x_numeric).float(),  # Shape: (16, 3)
            'y': torch.from_numpy(y_power).float(),  # Shape: (4,)
            'y_zenith': torch.from_numpy(y_zenith).float()  # Shape: (4,)
        }


if __name__ == "__main__":
    # 加载配置
    config = load_config("../config/config.yaml")
    # 测试代码
    csv_file = config["file_paths"]["series_file"]
    sat_dir = config["file_paths"]["aligned_satellite_path"]

    if os.path.exists(csv_file):
        ds = SatellitePVDataset(csv_file, sat_dir, mode='train')
        if len(ds) > 0:
            sample = ds[0]
            print(f"Input Image: {sample['x_images'].shape}")
            print(f"Input Numeric: {sample['x_numeric'].shape}")
            print(f"Target Power: {sample['y'].shape}")
            print(f"Target Zenith: {sample['y_zenith'].shape}")  # 测试新加的维度
    else:
        print("请先生成 CSV 文件")
