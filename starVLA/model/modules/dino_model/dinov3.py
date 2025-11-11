"""
DINOv3 vision backbone wrapper.

Features:
  - Loads DINOv3 ViT variants via torch.hub (with local fallback)
  - Exposes patch token features (x_norm_patchtokens)
  - Provides preprocessing (resize + crop + normalization) for multi-view PIL images
  - Parallel per-view preprocessing using ThreadPoolExecutor
"""

from collections import OrderedDict
import os

from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from torchvision import transforms


def apply_transform(view, transform):
    return transform(view)

# https://modelscope.cn/models/facebook/dinov3-vitl16-pretrain-lvd1689m
class DINOv3BackBone(nn.Module):
    """
    Thin wrapper around a DINOv3 ViT model.

    Args:
        backone_name: DINOv3 model id (e.g. dinov3_vits16, dinov3_vitl16).
    """

    def __init__(self, backone_name="dinov3_vitl16") -> None:
        super().__init__()
        try:
            self.body = torch.hub.load("facebookresearch/dinov3", backone_name)
        except Exception:
            import traceback

            traceback.print_exc()
            print(f"Failed to load {backone_name} from torch hub, loading from local")
            TORCH_HOME = os.environ.get("TORCH_HOME", "~/.cache/torch/")
            
            # 2. 权重和代码路径改为 dinov3
            # 注意：本地 hub 的文件夹名可能是 facebookresearch_dinov3_main
            code_path = os.path.expanduser(f"{TORCH_HOME}/hub/facebookresearch_dinov3_main")
            weights_path = os.path.expanduser(f"{TORCH_HOME}/hub/checkpoints/{backone_name}_pretrain.pth")
            
            self.body = torch.hub.load(code_path, backone_name, source="local", pretrained=False)
            
            state_dict = torch.load(weights_path)
            
            # torch.hub.load() 返回的已经是 ViT, state_dict 应该可以直接加载
            # 如果加载失败，可能需要检查 state_dict 的 keys (可能是 state_dict["vision_model"])
            self.body.load_state_dict(state_dict)

        # 3. (优化) 不再硬编码，直接从模型读取特征维度
        self.num_channels = self.body.embed_dim

        # 4. (优化) 使用 Resize + CenterCrop，防止图像变形
        self.dino_transform = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def forward(self, tensor):
        """
        Forward pass.

        Args:
            tensor: Image batch tensor [B*views, 3, H, W].

        Returns:
            torch.Tensor: Patch token features [B*views, N_tokens, C].
        """
        # 这个 API DINOv3 保持了和 DINOv2 一致
        # x_norm_patchtokens 是你从 DINOv3 ViT Backbone 中能拿到的最有用、最精华的特征输出，尤其是用于下游任务（如检测、分割）时。
        xs = self.body.forward_features(tensor)["x_norm_patchtokens"]
        return xs  # B*views, token, dim

    def prepare_dino_input(self, img_list):
        """
        Preprocess a batch of multi-view PIL image lists into a tensor suitable for DINO.

        Args:
            img_list: List of samples; each sample is List[PIL.Image] (multi-view).

        Returns:
            torch.Tensor: Flattened batch of shape [B * num_view, 3, H, W] on model device.
        """
        with ThreadPoolExecutor() as executor:
            image_tensors = torch.stack(
                [
                    torch.stack(list(executor.map(lambda view: apply_transform(view, self.dino_transform), views)))
                    for views in img_list
                ]
            )

        # move the tensor to the device of DINO encoder
        B, num_view, C, H, W = image_tensors.shape
        image_tensors = image_tensors.view(B * num_view, C, H, W)
        device = next(self.parameters()).device
        image_tensors = image_tensors.to(device)

        return image_tensors


def get_dino_v3_model(backone_name="dinov3_vitl16") -> DINOv3BackBone:
    """
    Factory helper returning a configured DINOv3BackBone.

    Args:
        backone_name: DINOv3 variant name.

    Returns:
        DINOv3BackBone: Initialized backbone instance.
    """
    return DINOv3BackBone(backone_name)


if __name__ == "__main__":
    # 测试 DINOv3 ViT-Base 模型
    print("Initializing DINOv3 ViT-Base Backbone...")
    dino_v3_b = get_dino_v3_model("dinov3_vitl16")
    print(f"Model: {dino_v3_b.__class__.__name__}")
    print(f"Feature dimension (num_channels): {dino_v3_b.num_channels}")

    # 创建一个 [Batch=2, Views=1] 的 dummy 图像数据
    # 每个 list 是一个 sample, list 里的 PIL 是 multi-views
    from PIL import Image
    dummy_image = Image.new('RGB', (400, 300), color = 'red')
    img_list = [
        [dummy_image],  # Sample 1, 1 view
        [dummy_image],  # Sample 2, 1 view
    ]

    # 预处理
    tensor = dino_v3_b.prepare_dino_input(img_list)
    print(f"Input tensor shape: {tensor.shape}") # 应该
    # 应该是 [2, 3, 224, 224]

    # 前向传播
    with torch.no_grad():
        features = dino_v3_b(tensor)
    
    print(f"Output features shape: {features.shape}") # 应该是 [2, 256, 768]
    print("Done.")