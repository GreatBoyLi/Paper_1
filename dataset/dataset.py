import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from utils.config import load_config


class SatellitePVDataset(Dataset):
    def __init__(self, csv_path, satellite_dir,
                 input_seq_len=16, output_seq_len=4,
                 mode='train', train_ratio=0.7, val_ratio=0.2):
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
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        if mode == 'train':
            self.data = self.df.iloc[:train_end]
        elif mode == 'val':
            self.data = self.df.iloc[train_end:val_end]
        elif mode == 'test':
            self.data = self.df.iloc[val_end:]
        else:
            raise ValueError(f"不支持的 mode: {mode}")

        # ==========================================
        # 严格的时间连续性校验
        # ==========================================
        self.valid_indices = []
        total_len = self.input_len + self.output_len
        expected_time_delta = pd.Timedelta(minutes=15 * (total_len - 1))

        max_possible_idx = len(self.data) - total_len
        for i in range(max_possible_idx + 1):
            start_time = self.data.index[i]
            end_time = self.data.index[i + total_len - 1]

            if end_time - start_time == expected_time_delta:
                self.valid_indices.append(i)

        print(f"[{mode}] 数据集加载完成 | 原始行数: {len(self.data)} | 严格连续的有效样本数: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    # ==========================================
    # 🌟 新增：动态获取安全的晴空背景温度
    # ==========================================
    def _get_safe_clear_sky_temp(self, timestamp):
        """
        根据时间戳判断南半球干旱区的季节和昼夜，返回合理的背景亮温兜底值 (K)。
        注意：此处假设 timestamp 为当地时间 (Local Time)。
        """
        month = timestamp.month
        hour = timestamp.hour

        # 1. 判断季节 (南半球)
        if month in [11, 12, 1, 2, 3]:
            season = 'summer'
        elif month in [5, 6, 7, 8, 9]:
            season = 'winter'
        else:
            season = 'transition'  # 春秋过渡季

        # 2. 判断昼夜 (粗略以早7点到晚18点为白天)
        is_day = 7 <= hour <= 18

        # 3. 返回对应的典型晴空地表亮温兜底值
        if season == 'summer':
            return 325.0 if is_day else 295.0
        elif season == 'winter':
            return 295.0 if is_day else 275.0
        else:
            return 310.0 if is_day else 285.0

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]

        hist_start = real_idx
        hist_end = real_idx + self.input_len
        future_end = hist_end + self.output_len

        # 获取数值特征和预测目标
        features = ['Power_Norm', 'Clear_Sky_GHI', 'Solar_Zenith']
        x_numeric = self.data.iloc[hist_start:hist_end][features].values
        y_power = self.data.iloc[hist_end:future_end]['Power_Norm'].values
        y_zenith = self.data.iloc[hist_end:future_end]['Solar_Zenith'].values

        # 获取图像数据
        hist_timestamps = self.data.index[hist_start:hist_end]
        images = []
        last_valid_img = None

        for ts in hist_timestamps:
            # 🌟 每次循环都根据当前时间戳获取动态兜底温度
            current_safe_temp = self._get_safe_clear_sky_temp(ts)

            file_name = f"sat_15min_{ts.strftime('%Y%m%d_%H%M')}.npy"
            yyyy, mm, dd = ts.strftime("%Y"), ts.strftime("%m"), ts.strftime("%d")
            yyyymm = f"{yyyy}{mm}"
            file_path = os.path.join(self.satellite_dir, yyyymm, dd, file_name)

            if os.path.exists(file_path):
                img = np.load(file_path).astype(np.float32)

                # 处理 NaN 和 Inf
                if np.isnan(img).any() or np.isinf(img).any():
                    valid_mean = np.nanmean(img[~np.isinf(img)])
                    if np.isnan(valid_mean):
                        img = last_valid_img if last_valid_img is not None else np.full(
                            (1, 96, 96), current_safe_temp, dtype=np.float32
                        )
                    else:
                        img = np.nan_to_num(img, nan=valid_mean, posinf=valid_mean, neginf=valid_mean)
                last_valid_img = img
            else:
                # 找不到文件时的全局兜底
                if last_valid_img is not None:
                    img = last_valid_img
                else:
                    img = np.full((1, 96, 96), current_safe_temp, dtype=np.float32)

            # ==========================================
            # 🌟 同步更新：扩大归一化范围，防止高温地表被截断
            # ==========================================
            img = (img - 180.0) / (345.0 - 180.0)
            img = np.clip(img, 0.0, 1.0)

            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=0)

            images.append(img)

        x_images = np.stack(images, axis=0)

        return {
            'x_images': torch.from_numpy(x_images).float(),
            'x_numeric': torch.from_numpy(x_numeric).float(),
            'y': torch.from_numpy(y_power).float(),
            'y_zenith': torch.from_numpy(y_zenith).float()
        }


if __name__ == "__main__":
    # 加载配置
    config = load_config("../config/config.yaml")
    csv_file = config["file_paths"]["series_file"]
    sat_dir = config["file_paths"]["aligned_satellite_path"]

    if os.path.exists(csv_file):
        ds = SatellitePVDataset(csv_file, sat_dir, mode='train')
        if len(ds) > 0:
            sample = ds[0]
            print(f"Input Image: {sample['x_images'].shape}")
            print(f"Input Numeric: {sample['x_numeric'].shape}")
            print(f"Target Power: {sample['y'].shape}")
            print(f"Target Zenith: {sample['y_zenith'].shape}")
    else:
        print("请先生成 CSV 文件")