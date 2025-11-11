# 文件名: starVLA/model/modules/map_model/__init__.py

import torch
from omegaconf import DictConfig

def get_map_model(config: DictConfig):
    """
    MapAnythingBackbone 的模型工厂。
    """
    
    # 导入 Backbone 封装器
    try:
        from .mapanything_backbone import MapAnythingBackbone
    except ImportError:
        print("错误：找不到 mapanything_backbone.py 文件。")
        print("请确保 mapanything_backbone.py 与此 __init__.py 在同一个文件夹中。")
        raise

    print("Loading original MapAnything model for backbone...")
    
    # 3. 使用 *原始* MapAnything 类的 *有效* from_pretrained 加载模型
    #    我们从 config 中读取 HF repo id
    model_repo_id = config.get("model_repo_id", "facebook/map-anything")
    
    model = MapAnythingBackbone.from_pretrained(
        model_repo_id,
        # force_download=True # 始终强制下载以避免缓存错误
    )
    
    print(f"MapAnythingBackbone initialized with {model_repo_id}")
    
    return model