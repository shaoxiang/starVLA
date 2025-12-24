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
    return transform(view_pil_image)

class MapAnythingBackbone(MapAnything):
    """
    MapAnything Backbone V2 (适配新版 MapAnything 库)
    保留空间结构 (H, W)，提取 Spatial Features 和 Metric Scale Token。
    """
    def __init__(self, image_size: Tuple[int, int] = (518, 518), *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.image_size = image_size
        print(f"[MapAnythingBackbone] 初始化: 输入分辨率设置为 {self.image_size}")
        
        # 覆盖/添加预处理器
        self.map_transform = transforms.Compose(
            [
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC), 
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def prepare_map_input(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将 VLA examples 转换为 MapAnything 需要的格式。
        """
        B = len(examples)
        if B == 0: return []
        
        V = len(examples[0]["image"]) 
        device = next(self.parameters()).device
        
        # 1. 处理图像
        batch_images_pil = [ex["image"] for ex in examples]
        views_as_batches = list(zip(*batch_images_pil))
        
        map_views_list = []
        with ThreadPoolExecutor() as executor:
            for view_batch_pils in views_as_batches:
                img_tensors = list(executor.map(
                    lambda img: _apply_transform(img, self.map_transform), 
                    view_batch_pils
                ))
                img_batch_tensor = torch.stack(img_tensors).to(device)
                
                # 新版 model.py 依赖 data_norm_type
                map_views_list.append({
                    "img": img_batch_tensor,
                    "data_norm_type": ["dinov2"] 
                })

        # 2. 处理几何输入
        # 这里的关键是确保 Tensor 维度正确：(B, ...)
        for i in range(V):
            # 处理 Intrinsics / Pose / Depth
            # 新版 model.py 通常期望 key 为 'camera_pose_quats', 'camera_pose_trans' 等
            # 这里做简化的透传，假设 examples 里的 key 已经被转换好，或者后续会被 validate
            pass
            
        # 简单的 key 映射逻辑 (根据你的 provided code)
        # 注意：model.py 对 key 的名字很敏感，例如 "depth_along_ray", "camera_pose_quats"
        for key in ["depth_along_ray", "is_metric_scale"]:
            if key in examples[0]:
                data_list = [ex[key] for ex in examples]
                if isinstance(data_list[0], torch.Tensor):
                    batch_tensor = torch.stack(data_list).to(device)
                else:
                    batch_tensor = torch.tensor(data_list, device=device)
                
                # 广播到每个 view
                for i in range(V):
                    map_views_list[i][key] = batch_tensor

        return map_views_list

    def forward(self, views: List[Dict[str, Any]], 
                memory_efficient_inference: bool = False) -> Dict[str, torch.Tensor]:
        """
        核心修改：适配 model.py 的 forward 流程
        """
        batch_size_per_view, _, height, width = views[0]["img"].shape
        
        # --- 1. Encode (新版返回 tuple) ---
        # model.py line 788: return all_encoder_features_across_views, all_encoder_registers_across_views
        all_encoder_features_across_views, all_encoder_registers_across_views = \
            self._encode_n_views(views)
        
        # --- 2. Fuse Geometry (必须只传 features list) ---
        # model.py 建议在 fusion 时关闭 autocast 以避免 NaN
        with torch.autocast("cuda", enabled=False):
            all_encoder_features_across_views = (
                self._encode_and_fuse_optional_geometric_inputs(
                    views, all_encoder_features_across_views
                )
            )
        
        # --- 3. Info Sharing (Transformer Input 构造方式变了) ---
        # 必须传入 scale token
        input_scale_token = (
            self.scale_token.unsqueeze(0).unsqueeze(-1)
            .repeat(batch_size_per_view, 1, 1)
        ) # [B, C, 1]

        # 构造输入对象，注意新增了 additional_input_tokens_per_view (registers)
        info_sharing_input = MultiViewTransformerInput(
            features=all_encoder_features_across_views,
            additional_input_tokens_per_view=all_encoder_registers_across_views,
            additional_input_tokens=input_scale_token,
        )

        # 执行 Transformer
        if self.info_sharing_return_type == "intermediate_features":
            final_feat, _ = self.info_sharing(info_sharing_input)
        else:
            final_feat = self.info_sharing(info_sharing_input)

        # --- 4. 提取输出 ---
        # final_feat.additional_token_features 是 Scale Token 输出
        # final_feat.features 是 List[Tensor] (每个 view 一个 [B, C, H, W])
        
        scale_token_output = final_feat.additional_token_features # [B, C, 1] 通常是这样，或者是 [B, 1, C]
        
        # 检查维度并调整，确保输出是 [B, 1, C]
        if scale_token_output.shape[-1] == 1 and scale_token_output.ndim == 3:
             scale_token_output = scale_token_output.permute(0, 2, 1) # [B, 1, C]
        
        # 堆叠所有 View 的特征
        stacked_patches = torch.stack(final_feat.features, dim=1) # [B, V, C, H, W]

        return {
            "spatial_features": stacked_patches,
            "metric_scale": scale_token_output
        }

if __name__ == "__main__":
    from PIL import Image
    print("=== MapAnything Backbone ===\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 初始化模型
    model = MapAnythingBackbone.from_pretrained("facebook/map-anything").to(device).eval()

    # 模拟数据：使用标准的 518x518
    B, H, W, num_views = 2, 518, 518, 2
    examples = [{
        "image": [Image.new('RGB', (W, H)) for _ in range(num_views)],
        "intrinsics": torch.stack([torch.eye(3)] * num_views),
        "is_metric_scale": True
    } for _ in range(B)]

    try:
        inputs = model.prepare_map_input(examples)
        with torch.no_grad():
            output = model(inputs)
        
        print(f"Success!")
        print(f"Spatial Features: {output['spatial_features'].shape}") 
        print(f"Metric Scale:     {output['metric_scale'].shape}")
    except Exception:
        import traceback
        traceback.print_exc()