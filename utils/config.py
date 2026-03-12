import os
import yaml
import logging
import datetime


def load_config(config_path):
    """加载YAML配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ 配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)  # safe_load避免安全风险
    return config


def setup_logger(save_dir):
    # 确保文件夹存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 生成类似 train_20260312_103000.log 的文件名
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"train_{current_time}.log")

    # 配置 logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',  # 自动加上时间戳
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # 写入文件
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    return logging.getLogger(__name__)
