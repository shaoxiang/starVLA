# 文件名: mapanything_backbone.py

import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from torchvision import transforms

try:
    from mapanything.models import MapAnything
    from uniception.models.info_sharing.base import MultiViewTransformerInput
except ImportError:
    print("错误：请确保你已经安装了 map-anything 库。")
    exit()

def _apply_transform(view_pil_image, transform):
    """用于 ThreadPoolExecutor 的顶层辅助函数"""
    return transform(view_pil_image)

class MapAnythingBackbone(MapAnything):
    """
    一个 MapAnything 的“Backbone”封装器。
    保留空间结构 (H, W)，便于下游进行准确的几何计算。
    """
    def __init__(self, image_size: Tuple[int, int] = (518, 518), *args, **kwargs):
        """
        初始化函数。
        
        它会调用 MapAnything 的原始 __init__ 方法，
        并额外创建图像预处理器。
        Args:
            image_size: (H, W). MapAnything 原生训练尺寸为 518x518。
                        显存允许时强烈建议使用 518，否则几何细节会丢失。
        """
        super().__init__(*args, **kwargs)

        self.image_size = image_size
        print(f"[MapAnythingBackbone] 初始化: 输入分辨率设置为 {self.image_size}")
        
        # --- 预处理器现在是类的一部分 ---
        # MapAnything 的 DINOv2 期望 224x224, Resize+CenterCrop

        # --- 🚀 速度优化核心 ---
        # 原始 MapAnything 使用 BICUBIC，这在训练时太慢了
        self.map_transform = transforms.Compose(
            [
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC), 
                transforms.CenterCrop(self.image_size), # 确保是正方形
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def prepare_map_input(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将 (B,) 的 VLA `examples` 字典列表，转换为 MapAnything.forward 所需的 
        (V,) 字典列表 (其中每个字典包含 [B, ...] 的张量)。
        
        Args:
            examples: List[Dict[str, Any]], 你的 VLA 批次数据。
                      - 必需: examples[i]["image"] -> List[PIL.Image] (多视角)
                      - 可选: examples[i]["intrinsics"] -> torch.Tensor [V, 3, 3]
                      - 可选: examples[i]["camera_poses"] -> torch.Tensor [V, 4, 4]
                      - 可选: examples[i]["depth_along_ray"] -> torch.Tensor [V, H, W, 1]
                      - 可选: examples[i]["is_metric_scale"] -> bool

        Returns:
            List[Dict[str, Any]], 形状 (NumViews, 
                {"img": Tensor[B, C, H, W], "data_norm_type": ["dinov2"], ...})
        """
        B = len(examples)
        if B == 0:
            return []
        
        # 假设所有样本的视角数都相同
        V = len(examples[0]["image"]) 
        device = next(self.parameters()).device
        
        # --- 1. 处理图像 (必需) ---
        batch_images_pil = [ex["image"] for ex in examples] # (B, V) 的 PIL 列表
        views_as_batches = list(zip(*batch_images_pil))    # (V, B) 的 PIL 列表
        
        map_views_list = []
        with ThreadPoolExecutor() as executor:
            for view_batch_pils in views_as_batches:
                # view_batch_pils 是一个 tuple (B,)，包含一个视图的所有 batch 图像
                # 并行应用 transform
                img_tensors = list(executor.map(
                    lambda img: _apply_transform(img, self.map_transform), 
                    view_batch_pils
                ))
                
                img_batch_tensor = torch.stack(img_tensors).to(device)
                
                map_views_list.append({
                    "img": img_batch_tensor,
                    "data_norm_type": ["dinov2"] # 必须是列表!
                })

        # --- 2. 处理可选的几何输入 (可扩展) ---
        for key in ["intrinsics", "camera_poses", "depth_along_ray"]:
            if key in examples[0]:
                try:
                    batch_tensors = torch.stack([ex[key] for ex in examples])
                    # (B, V, ...) -> (V, B, ...)
                    v_first = batch_tensors.permute(1, 0, *range(2, batch_tensors.dim()))
                    for i in range(V): map_views_list[i][key] = v_first[i].to(device)
                except: pass
        
        if "is_metric_scale" in examples[0]:
            batch_data = [ex["is_metric_scale"] for ex in examples]
            batch_tensor = torch.tensor(batch_data, dtype=torch.bool, device=device)
            for i in range(V): map_views_list[i]["is_metric_scale"] = batch_tensor

        return map_views_list


    def forward(self, views: List[Dict[str, Any]], 
                memory_efficient_inference: bool = False) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict: {
                'patch_features': [B, V, H_p, W_p, C], 
                'scale_token': [B, 1, C]
            }
        """
        
        # --- 这部分代码 1:1 复制自 MapAnything.forward ---
        batch_size_per_view, _, height, width = views[0]["img"].shape
        # num_views = len(views)

        # 1. Encode
        all_encoder_features_across_views = self._encode_n_views(views)
        
        # 2. Fuse Geometry (Optional)
        all_encoder_features_across_views = (
            self._encode_and_fuse_optional_geometric_inputs(
                views, all_encoder_features_across_views
            )
        )
        
        # 3. Info Sharing (Transformer)
        input_scale_token = (
            self.scale_token.unsqueeze(0).unsqueeze(-1)
            .repeat(batch_size_per_view, 1, 1)
        )

        info_sharing_input = MultiViewTransformerInput(
            features=all_encoder_features_across_views,
            additional_input_tokens=input_scale_token,
        )

        if self.info_sharing_return_type == "intermediate_features":
            final_feat, _ = self.info_sharing(info_sharing_input)
        else:
            final_feat = self.info_sharing(info_sharing_input)

        # --- 原始 forward 到此结束，开始我们自己的提取 ---
        # --- 保留 (H, W) 结构 ---
        scale_token_output = final_feat.additional_token_features.permute(0, 2, 1) # [B, 1, C]
        
        # patch_tokens_list 是一个 list，长度为 V，每个元素是 [B, C, H, W]
        # 我们将其堆叠
        stacked_patches = torch.stack(final_feat.features, dim=1) # [B, V, C, H, W]

        return {
            "spatial_features": stacked_patches,
            "metric_scale": scale_token_output
        }
    

if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    
    print("=== Mapanything backbone (V2 - 封装了预处理) ===\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading original MapAnything model...")
    model = MapAnythingBackbone.from_pretrained(
        "facebook/map-anything",
    ).to(device)
    
    # 验证 transform 是否已存在
    assert hasattr(model, 'map_transform'), "模型缺少 'map_transform' 属性!"
    print("模型初始化完成 (包含 'map_transform')")

    print("\nTesting backbone forward pass with new prepare_map_input...")
    
    B, H, W = 2, 504, 504
    C_info_sharing = model.info_sharing.dim 
    num_views = 2
    num_patches_per_view = (H // 14) * (W // 14) # 36*36 = 1296
    
    # --- 1. 创建 VLA 风格的 `examples` 批次 ---
    examples = []
    for _ in range(B):
        examples.append({
            "image": [
                Image.new('RGB', (W, H), color='red'), # 视图 1
                Image.new('RGB', (W, H), color='blue') # 视图 2
            ],
            # --- 2. 添加可选的几何信息 ---
            "intrinsics": torch.stack([torch.eye(3)] * num_views), # (V, 3, 3)
            "camera_poses": torch.stack([torch.eye(4)] * num_views), # (V, 4, 4)
            "is_metric_scale": True
        })

    try:
        # 3. 调用新的 prepare_map_input 方法
        # 它现在是模型的一个方法
        print("Running model.prepare_map_input(examples)...")
        views_list = model.prepare_map_input(examples)
        
        print(f"预处理完成。生成了 {len(views_list)} 个视图的输入。")
        print(f"视图 0 'img' 张量形状: {views_list[0]['img'].shape}")
        print(f"视图 0 'intrinsics' 张量形状: {views_list[0]['intrinsics'].shape}")
        
        # 验证 B 维度是否正确
        assert views_list[0]['img'].shape[0] == B
        assert views_list[0]['intrinsics'].shape[0] == B
        
        with torch.no_grad():
            map_output = model(views_list)

        patch_tokens = map_output["spatial_features"] # [B, V, C, H, W]
        scale_token = map_output["metric_scale"]      # [B, 1, C]

        print("\n--- 🚀 Success! ---")
        print(f"3D Patch Tokens Shape: {patch_tokens.shape}")
        print(f"3D Scale Token Shape: {scale_token.shape}")
        
        expected_patch_shape = (B, num_views * num_patches_per_view, C_info_sharing)
        expected_scale_shape = (B, 1, C_info_sharing)

        print(f"Expected Patch Tokens: {expected_patch_shape}")
        print(f"Expected Scale Token:  {expected_scale_shape}")
        
        assert patch_tokens.shape == expected_patch_shape
        assert scale_token.shape == expected_scale_shape
        print("\nToken shapes match expected output.")

    except Exception as e:
        print(f"\n--- ❌ Error during forward pass ---")
        import traceback
        traceback.print_exc()