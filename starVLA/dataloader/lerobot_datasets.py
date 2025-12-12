# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025]. 
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025]. 
# Modification: [suport topdowm processing, suport param from config].

from pathlib import Path
from typing import Sequence
from omegaconf import OmegaConf
import torch
import numpy as np

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG, EmbodimentTag

def collate_fn(batch):
    return batch

def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :return: A LeRobotSingleDataset object.
    """
    
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
    
    video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "decord"
    
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend, # decord is more efficiency | torchvision_av for video.av1
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
    )

def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    delete_pause_frame: bool = True,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:  
        dataset_key = (d_name, robot_type)
        print(f"Considering Dataset: `{(d_name, d_weight, robot_type)}`")
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append((make_LeRobotSingleDataset(Path(data_root_dir), d_name, robot_type, delete_pause_frame=delete_pause_frame, data_cfg=data_cfg), d_weight))

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )

def print_batch_details(batch, max_depth=3, current_depth=0):
    """
    递归打印 batch 的结构、内容和形状。
    
    Args:
        batch: 要打印的 batch 数据
        max_depth: 最大递归深度
        current_depth: 当前递归深度
    """
    indent = "  " * current_depth
    
    if current_depth > max_depth:
        print(f"{indent}... (深度超过 {max_depth})")
        return
    
    if isinstance(batch, list):
        print(f"{indent}List with {len(batch)} items:")
        for i, item in enumerate(batch):
            print(f"{indent}  Item {i}:")
            print_batch_details(item, max_depth, current_depth + 1)
            if i >= 2:  # 只打印前3个项目
                print(f"{indent}  ... (还有 {len(batch) - 3} 个项目)")
                break
    elif isinstance(batch, dict):
        print(f"{indent}Dict with keys: {list(batch.keys())}")
        for key, value in batch.items():
            print(f"{indent}  Key '{key}':")
            print_batch_details(value, max_depth, current_depth + 1)
    elif isinstance(batch, (torch.Tensor, np.ndarray)):
        print(f"{indent}  Shape: {batch.shape}, Dtype: {batch.dtype}")
        numel = batch.numel() if isinstance(batch, torch.Tensor) else batch.size
        if numel <= 10:  # 小张量打印值
            print(f"{indent}  Values: {batch}")
        else:
            print(f"{indent}  Values (first 5): {batch.flatten()[:5]}")
    elif isinstance(batch, str):
        print(f"{indent}  String: '{batch}' (length: {len(batch)})")
    elif isinstance(batch, (int, float)):
        print(f"{indent}  Value: {batch}")
    else:
        print(f"{indent}  Type: {type(batch)}, Value: {str(batch)[:100]}...")

if __name__ == "__main__":

    # import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_behavior.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # debugpy.listen(("0.0.0.0", 10092))
    # print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    # debugpy.wait_for_client()

    # args.config_yaml = "examples/SimplerEnv/train_files/starvla_cotrain_libero.yaml"
    args.config_yaml = "examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml"
    cfg = OmegaConf.load(args.config_yaml)

    vla_dataset_cfg = cfg.datasets.vla_data
    # vla_dataset_cfg.data_root_dir = "./playground/Datasets/behavior-1k"
    # vla_dataset_cfg.include_state = True
    # vla_dataset_cfg.data_mix = "BEHAVIOR_dual_base_depth"
    vla_dataset_cfg.task_id = 1
    # for task_id in ["all"]:
    for task_id in [5,11,13,26,36,27,43,44,45,46]:
        # 11,26,36,37
        # 5,11,13,26,36,27,43,44,45,46
        # 2,3,5,11,13,25,26,27,
        # 3,5,11,13, / 14,15,16,17, / 19,20,23,25, / 26,27,30,34, / 36,37,38,39, 41,42,43,44,45,46,47,49
        vla_dataset_cfg.task_id = task_id
        print(f"Testing Task ID: {task_id}")
        dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        # dataset
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1, # For Debug
        collate_fn=collate_fn,
    )

    from tqdm import tqdm
    count = 1
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        print(batch)
        print(f"\n=== Batch {count} ===")
        print_batch_details(batch)
        if count >= 2:  # 只打印前2个batch
            break
        count += 1