"""
统一的DINOv2和DINOv3视觉骨干网络实现

Features:
  - 支持DINOv2和DINOv3两种模型变体
  - 统一的API接口
  - 加载模型通过torch.hub（支持本地回退）
  - 提供预处理功能（resize + 归一化）用于多视角PIL图像
  - 使用ThreadPoolExecutor进行并行预处理
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


class DINOBackBone(nn.Module):
    """
    统一的DINOv2和DINOv3视觉骨干网络包装器

    Args:
        backbone_name: DINO模型名称 (e.g. dinov2_vits14, dinov2_vitb14, dinov3_vits16, dinov3_vitl16).
        model_version: 模型版本 ('v2' 或 'v3')，如果为None则自动推断
    """

    def __init__(self, backbone_name="dinov2_vits14", model_version=None) -> None:
        super().__init__()
        
        # 自动推断模型版本
        if model_version is None:
            if "dinov2" in backbone_name.lower():
                model_version = "v2"
            elif "dinov3" in backbone_name.lower():
                model_version = "v3"
            else:
                raise ValueError(f"无法从backbone_name '{backbone_name}' 推断模型版本，请指定model_version参数")
        
        self.model_version = model_version.lower()
        
        # 加载模型
        try:
            if self.model_version == "v2":
                self.body = torch.hub.load("facebookresearch/dinov2", backbone_name)
            elif self.model_version == "v3":
                self.body = torch.hub.load("facebookresearch/dinov3", backbone_name)
            else:
                raise ValueError(f"不支持的模型版本: {model_version}")
        except Exception:
            import traceback
            traceback.print_exc()
            
            print(f"从torch hub加载{backbone_name}失败，尝试从本地加载")
            TORCH_HOME = os.environ.get("TORCH_HOME", "~/.cache/torch/")
            
            if self.model_version == "v2":
                code_path = os.path.expanduser(f"{TORCH_HOME}/hub/facebookresearch_dinov2_main")
            elif self.model_version == "v3":
                code_path = os.path.expanduser(f"{TORCH_HOME}/hub/facebookresearch_dinov3_main")
            
            weights_path = os.path.expanduser(f"{TORCH_HOME}/hub/checkpoints/{backbone_name}_pretrain.pth")
            
            self.body = torch.hub.load(code_path, backbone_name, source="local", pretrained=False)
            
            state_dict = torch.load(weights_path)
            self.body.load_state_dict(state_dict)

        # 根据模型版本和名称设置特征维度
        if self.model_version == "v2":
            if backbone_name == "dinov2_vits14":
                self.num_channels = 384
            elif backbone_name == "dinov2_vitb14":
                self.num_channels = 768
            elif backbone_name == "dinov2_vitl14":
                self.num_channels = 1024
            elif backbone_name == "dinov2_vitg14":
                self.num_channels = 1408
            else:
                # 尝试从模型直接获取特征维度
                try:
                    self.num_channels = self.body.embed_dim
                except AttributeError:
                    raise NotImplementedError(f"DINOv2 backbone {backbone_name} not implemented")
        elif self.model_version == "v3":
            # DINOv3 可以直接从模型获取特征维度
            self.num_channels = self.body.embed_dim

        # 根据模型版本设置预处理
        if self.model_version == "v2":
            # DINOv2使用固定尺寸resize
            self.dino_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        elif self.model_version == "v3":
            # DINOv3使用Resize + CenterCrop防止图像变形
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
        前向传播

        Args:
            tensor: 图像批次张量 [B*views, 3, H, W].

        Returns:
            torch.Tensor: Patch token特征 [B*views, N_tokens, C].
        """
        features = self.body.forward_features(tensor)
        
        # 两个版本都使用相同的输出键
        if "x_norm_patchtokens" in features:
            xs = features["x_norm_patchtokens"]
        else:
            # 如果没有x_norm_patchtokens，则返回最后一层的特征
            xs = features["x_norm_clstoken"].unsqueeze(1)  # 添加序列维度
        
        return xs  # B*views, token, dim

    def prepare_dino_input(self, img_list):
        """
        预处理多视角PIL图像列表为适合DINO的张量

        Args:
            img_list: 样本列表；每个样本是List[PIL.Image] (多视角).

        Returns:
            torch.Tensor: 展平批次，形状为 [B * num_view, 3, H, W]，在模型设备上.
        """
        with ThreadPoolExecutor() as executor:
            image_tensors = torch.stack(
                [
                    torch.stack(list(executor.map(lambda view: apply_transform(view, self.dino_transform), views)))
                    for views in img_list
                ]
            )

        # 将张量移到模型设备
        B, num_view, C, H, W = image_tensors.shape
        image_tensors = image_tensors.view(B * num_view, C, H, W)
        device = next(self.parameters()).device
        image_tensors = image_tensors.to(device)

        return image_tensors


def get_dino_model(backbone_name="dinov2_vits14", model_version=None) -> DINOBackBone:
    """
    工厂函数，返回配置好的DINOBackBone实例

    Args:
        backbone_name: DINO模型变体名称
        model_version: 模型版本 ('v2' 或 'v3')，如果为None则自动推断

    Returns:
        DINOBackBone: 初始化的骨干网络实例
    """
    return DINOBackBone(backbone_name, model_version)


def get_dino_v2_model(backbone_name="dinov2_vits14") -> DINOBackBone:
    """
    工厂函数，返回配置好的DINOv2骨干网络实例

    Args:
        backbone_name: DINOv2模型变体名称

    Returns:
        DINOBackBone: 初始化的DINOv2骨干网络实例
    """
    return DINOBackBone(backbone_name, model_version="v2")


def get_dino_v3_model(backbone_name="dinov3_vitl16") -> DINOBackBone:
    """
    工厂函数，返回配置好的DINOv3骨干网络实例

    Args:
        backbone_name: DINOv3模型变体名称

    Returns:
        DINOBackBone: 初始化的DINOv3骨干网络实例
    """
    return DINOBackBone(backbone_name, model_version="v3")


if __name__ == "__main__":
    print("=== 测试统一的DINO骨干网络实现 ===\n")
    
    # 测试DINOv2 ViT-S/14模型
    print("1. 初始化DINOv2 ViT-S/14骨干网络...")
    dino_v2_s = get_dino_v2_model("dinov2_vits14")
    print(f"   模型: {dino_v2_s.__class__.__name__}")
    print(f"   特征维度 (num_channels): {dino_v2_s.num_channels}")
    print(f"   模型版本: {dino_v2_s.model_version}")
    
    # 测试DINOv3 ViT-L/16模型
    print("\n2. 初始化DINOv3 ViT-L/16骨干网络...")
    dino_v3_l = get_dino_v3_model("dinov3_vitl16")
    print(f"   模型: {dino_v3_l.__class__.__name__}")
    print(f"   特征维度 (num_channels): {dino_v3_l.num_channels}")
    print(f"   模型版本: {dino_v3_l.model_version}")

    # 测试DINOv3 ViT-S/16模型
    print("\n2. 初始化DINOv3 ViT-S/16骨干网络...")
    dino_v3_s = get_dino_v3_model("dinov3_vits16")
    print(f"   模型: {dino_v3_s.__class__.__name__}")
    print(f"   特征维度 (num_channels): {dino_v3_s.num_channels}")
    print(f"   模型版本: {dino_v3_s.model_version}")
    
    # 测试自动推断模型版本
    print("\n3. 使用自动推断版本加载模型...")
    dino_auto_v2 = get_dino_model("dinov2_vitb14")
    print(f"   自动推断DINOv2: {dino_auto_v2.model_version}, 特征维度: {dino_auto_v2.num_channels}")
    
    dino_auto_v3 = get_dino_model("dinov3_vits16")
    print(f"   自动推断DINOv3: {dino_auto_v3.model_version}, 特征维度: {dino_auto_v3.num_channels}")
    
    # 创建测试图像
    from PIL import Image
    dummy_image = Image.new('RGB', (400, 300), color='red')
    img_list = [
        [dummy_image],  # 样本1, 1视角
        [dummy_image],  # 样本2, 1视角
    ]
    
    # 测试DINOv2预处理和前向传播
    print("\n4. 测试DINOv2预处理和前向传播...")
    tensor_v2 = dino_v2_s.prepare_dino_input(img_list)
    print(f"   输入张量形状: {tensor_v2.shape}")  # 应该是 [2, 3, 224, 224]
    
    with torch.no_grad():
        features_v2 = dino_v2_s(tensor_v2)
    
    print(f"   输出特征形状: {features_v2.shape}")  # 应该是 [2, 256, 384] 或类似
    print("   DINOv2测试完成")
    
    # 测试DINOv3预处理和前向传播
    print("\n5. 测试DINOv3预处理和前向传播...")
    tensor_v3 = dino_v3_l.prepare_dino_input(img_list)
    print(f"   输入张量形状: {tensor_v3.shape}")  # 应该是 [2, 3, 224, 224]
    
    with torch.no_grad():
        features_v3 = dino_v3_l(tensor_v3)
    
    print(f"   输出特征形状: {features_v3.shape}")  # 应该是 [2, 256, 1024] 或类似
    print("   DINOv3测试完成")
    
    print("\n=== 所有测试完成 ===")