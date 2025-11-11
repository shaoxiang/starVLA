# 文件名: mapanything_backbone.py
# (请确保你的文件名不是 mapanything.py，以免和库冲突)

import torch
import torch.nn as nn
from typing import List, Dict, Any

# --- 关键的原始库导入 ---
try:
    from mapanything.models import MapAnything
    from uniception.models.info_sharing.base import MultiViewTransformerInput
except ImportError:
    print("错误：请确保你已经安装了 map-anything 库。")
    print("pip install map-anything")
    exit()


class MapAnythingBackbone(MapAnything):
    """
    一个 MapAnything 的“Backbone”封装器。

    这个类继承自 MapAnything，但重写了 forward 方法，
    使其在 Multi-View Transformer 之后立即返回融合后的
    3D patch tokens 和 3D scale token，
    而不是返回最终的 3D 重建结果。

    这使它能像 DINOv3 Backbone 一样，为下游任务（如 VLA）
    提供丰富的 2D+3D 特征。
    """

    def __init__(self, *args, **kwargs):
        """
        初始化函数。
        
        它会调用 MapAnything 的原始 __init__ 方法，
        因此加载预训练权重等一切操作都和原来一样。
        """
        super().__init__(*args, **kwargs)

    def forward(self, views: List[Dict[str, Any]], 
                memory_efficient_inference: bool = False) -> (torch.Tensor, torch.Tensor):
        """
        重写的 Forward 传播。

        执行到 Multi-View Transformer，然后提取特征并返回。

        Args:
            views (List[dict]): 视图列表，与 MapAnything.forward 相同
            memory_efficient_inference (bool): (此实现中未使用，但保留签名一致性)

        Returns:
            all_patch_tokens (torch.Tensor): 
                形状为 [B, N*NumPatches, C] 的 3D Patch Tokens。
                (例如 [B, 2*196, 768])
                N 是视图数量, NumPatches 是每个视图的 patch 数, C 是特征维度。
            
            scale_token_output (torch.Tensor): 
                形状为 [B, 1, C] 的 3D Scale Token。
                (例如 [B, 1, 768])
        """
        
        # --- 这部分代码 1:1 复制自 MapAnything.forward ---
        # --- 目的是为了得到 Multi-View Transformer 的输入 ---
        
        # Get input shape of the images, number of views, and batch size per view
        batch_size_per_view, _, height, width = views[0]["img"].shape
        num_views = len(views)

        # Run the image encoder on all the input views
        all_encoder_features_across_views = self._encode_n_views(views)

        # Encode the optional geometric inputs and fuse with the encoded features from the N input views
        with torch.autocast("cuda", enabled=False):
            all_encoder_features_across_views = (
                self._encode_and_fuse_optional_geometric_inputs(
                    views, all_encoder_features_across_views
                )
            )

        # Expand the scale token to match the batch size
        input_scale_token = (
            self.scale_token.unsqueeze(0)
            .unsqueeze(-1)
            .repeat(batch_size_per_view, 1, 1)
        )  # (B, C, 1)

        # Combine all images into view-centric representation
        info_sharing_input = MultiViewTransformerInput(
            features=all_encoder_features_across_views,
            additional_input_tokens=input_scale_token,
        )
        
        # --- 这是最关键的一步：执行 Multi-View Transformer ---
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

        # 1. 提取 3D Scale Token
        # 原始形状是 [B, C, 1]，我们 permute 成 [B, 1, C]
        scale_token_output = final_info_sharing_multi_view_feat.additional_token_features.permute(0, 2, 1)
        
        # 2. 提取 3D Patch Tokens
        # .features 是一个列表，包含 N 个视图的特征
        # 每个张量形状为 [B, C, H_patch, W_patch]
        patch_tokens_list = final_info_sharing_multi_view_feat.features
        
        flattened_patch_tokens = []
        for view_tokens in patch_tokens_list:
            # view_tokens 形状: [B, C, H_p, W_p]
            B, C, H_p, W_p = view_tokens.shape
            
            # 1. flatten 空间维度: [B, C, H_p*W_p]
            view_tokens_flat = view_tokens.flatten(2)
            
            # 2. permute 维度: [B, H_p*W_p, C] (例如 [B, 196, 768])
            view_tokens_flat_permuted = view_tokens_flat.permute(0, 2, 1)
            
            flattened_patch_tokens.append(view_tokens_flat_permuted)
            
        # 3. 沿着“patch 数量”的维度拼接所有视图的 tokens
        # 得到 [B, N*(H_p*W_p), C] (例如 [B, 2*196, 768])
        all_patch_tokens = torch.cat(flattened_patch_tokens, dim=1)

        # 返回你想要的两个 Token 张量！
        return all_patch_tokens, scale_token_output
    

if __name__ == "__main__":
    print("=== Mapanything backbone ===\n")
    # Get inference device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 关键修复 ---
    # 1. 使用 *原始* MapAnything 类的 *有效* from_pretrained 加载模型
    print("Loading original MapAnything model...")
    model = MapAnythingBackbone.from_pretrained(
        "facebook/map-anything"
    ).to(device)

    # 2. 猴子补丁！
    # print("Patching class to MapAnythingBackbone...")
    # model.__class__ = MapAnythingBackbone
    
    # --- 修复结束 ---

    print(f"Model is now type: {type(model)}")
    print("模型初始化完成！")

    # 3. 添加一个测试来验证它是否有效
    print("\nTesting backbone forward pass...")
    
    # MapAnything / DINOv2 ViT-L patch_size=14
    # 我们用 504x504 的图像 (504 / 14 = 36 patches)
    B = 1
    H, W = 504, 504
    # Multi-View Transformer 的维度是 768
    C_info_sharing = 768 
    
    view1_img = torch.rand(B, 3, H, W).to(device)
    view2_img = torch.rand(B, 3, H, W).to(device)
    num_views = 2
    num_patches_per_view = (H // 14) * (W // 14) # 36*36 = 1296

    # --- 关键修正如下 ---
    # 1. "data_norm_type" 必须是一个列表 (list)
    # 2. 列表中的字符串必须是 "dinov2"，正如报错信息所提示
    views_list = [
        {"img": view1_img, "data_norm_type": ["dinov2"]},
        {"img": view2_img, "data_norm_type": ["dinov2"]}
    ]
    # --- 修正结束 ---

    try:
        with torch.no_grad():
            # 这现在将调用 MapAnythingBackbone.forward()
            patch_tokens, scale_token = model(views_list)
        
        print("\n--- 🚀 Success! ---")
        print(f"3D Patch Tokens Shape: {patch_tokens.shape}")
        print(f"3D Scale Token Shape: {scale_token.shape}")

        # 3D Patch Tokens Shape: torch.Size([1, 2592, 768])
        # 3D Scale Token Shape: torch.Size([1, 1, 768])
        # Expected Patch Tokens: (1, 2592, 768)
        # Expected Scale Token:  (1, 1, 768)
        
        # 查找 info_sharing_config 的 embed_dim (通常是 768)
        # 注意：DINOv2 ViT-L 是 1024，但 info_sharing transformer 是 768
        # 我们可以从加载的模型中动态获取它
        C_info_sharing = model.info_sharing.dim 
        
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