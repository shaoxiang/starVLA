# 文件名: mapanything_backbone.py

import torch
import torch.nn as nn
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from torchvision import transforms

# --- 关键的原始库导入 ---
try:
    from mapanything.models import MapAnything
    from uniception.models.info_sharing.base import MultiViewTransformerInput
except ImportError:
    print("错误：请确保你已经安装了 map-anything 库。")
    print("pip install map-anything")
    exit()

def _apply_transform(view_pil_image, transform):
    """用于 ThreadPoolExecutor 的顶层辅助函数"""
    return transform(view_pil_image)

class MapAnythingBackbone(MapAnything):
    """
    一个 MapAnything 的“Backbone”封装器。

    这个类继承自 MapAnything，但重写了 forward 方法，
    使其在 Multi-View Transformer 之后立即返回融合后的
    3D patch tokens 和 3D scale token。

    它还包含了完整的预处理逻辑 `prepare_map_input`。
    改进：支持自定义分辨率，默认为 518 (最佳几何性能)。
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
        
        # 辅助函数：从 (B, V, ...) 转换为 (V, B, ...)
        def _transpose_and_inject(key_name: str, tensor_key: str):
            if key_name in examples[0]:
                try:
                    # 1. 堆叠批次: (B, V, ...)
                    batch_tensors = torch.stack([ex[key_name] for ex in examples])
                    # 2. 交换 B 和 V 维度: (V, B, ...)
                    v_first_tensors = batch_tensors.permute(1, 0, *range(2, batch_tensors.dim()))
                    # 3. 注入到 map_views_list
                    for i in range(V):
                        map_views_list[i][tensor_key] = v_first_tensors[i].to(device)
                except Exception as e:
                    print(f"警告: 未能处理可选输入 '{key_name}'. 错误: {e}")

        # 辅助函数：处理 B-dim 的 bool/tensor
        def _inject_per_batch(key_name: str, tensor_key: str):
             if key_name in examples[0]:
                try:
                    # 1. 收集批次: (B,)
                    batch_data = [ex[key_name] for ex in examples]
                    if isinstance(batch_data[0], bool):
                         batch_tensor = torch.tensor(batch_data, dtype=torch.bool, device=device)
                    else:
                         batch_tensor = torch.stack(batch_data).to(device)
                    
                    # 2. 注入到 *每个* 视图 (MapAnything 期望 B-dim 的信息在每个视图中都存在)
                    for i in range(V):
                        map_views_list[i][tensor_key] = batch_tensor
                except Exception as e:
                    print(f"警告: 未能处理可选输入 '{key_name}'. 错误: {e}")

        
        # MapAnything.infer 接受 'intrinsics'
        #
        _transpose_and_inject("intrinsics", "intrinsics")
        
        # MapAnything.infer 接受 'camera_poses'
        #
        _transpose_and_inject("camera_poses", "camera_poses")

        # MapAnything.infer 接受 'depth_z' (来自 'depth_along_ray')
        #
        _transpose_and_inject("depth_along_ray", "depth_along_ray") 
        
        # MapAnything.infer 接受 'is_metric_scale'
        #
        _inject_per_batch("is_metric_scale", "is_metric_scale")

        return map_views_list


    def forward(self, views: List[Dict[str, Any]], 
                memory_efficient_inference: bool = False) -> (torch.Tensor, torch.Tensor):
        """
        重写的 Forward 传播。
        """
        
        # --- 这部分代码 1:1 复制自 MapAnything.forward ---
        batch_size_per_view, _, height, width = views[0]["img"].shape
        num_views = len(views)
        all_encoder_features_across_views = self._encode_n_views(views)
        
        # with torch.autocast("cuda", enabled=False):

        all_encoder_features_across_views = (
            self._encode_and_fuse_optional_geometric_inputs(
                views, all_encoder_features_across_views
            )
        )
        
        input_scale_token = (
            self.scale_token.unsqueeze(0)
            .unsqueeze(-1)
            .repeat(batch_size_per_view, 1, 1)
        )
        info_sharing_input = MultiViewTransformerInput(
            features=all_encoder_features_across_views,
            additional_input_tokens=input_scale_token,
        )
        if self.info_sharing_return_type == "no_intermediate_features":
            final_info_sharing_multi_view_feat = self.info_sharing(info_sharing_input)
        elif self.info_sharing_return_type == "intermediate_features":
            (
                final_info_sharing_multi_view_feat,
                intermediate_info_sharing_multi_view_feat,
            ) = self.info_sharing(info_sharing_input)
        else:
            raise ValueError(f"Invalid info_sharing_return_type: {self.info_sharing_return_type}")
        # --- 原始 forward 到此结束，开始我们自己的提取 ---
        scale_token_output = final_info_sharing_multi_view_feat.additional_token_features.permute(0, 2, 1)
        patch_tokens_list = final_info_sharing_multi_view_feat.features
        flattened_patch_tokens = []
        for view_tokens in patch_tokens_list:
            B, C, H_p, W_p = view_tokens.shape
            view_tokens_flat = view_tokens.flatten(2)
            view_tokens_flat_permuted = view_tokens_flat.permute(0, 2, 1)
            flattened_patch_tokens.append(view_tokens_flat_permuted)
        all_patch_tokens = torch.cat(flattened_patch_tokens, dim=1)
        
        return all_patch_tokens, scale_token_output
    

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
            patch_tokens, scale_token = model(views_list)
        
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