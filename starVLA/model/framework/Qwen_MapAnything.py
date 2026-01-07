# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0.
# Optimized by Gemini based on user's specific robot state definition.

"""
QwenSuper-GeometricConstraint Framework (State-Aware Version)
核心改进：
1. 深度适配 State [x,y,z,r,p,y,g] 结构
2. 利用 MapAnything 的 Metric Scale 实现视觉-物理空间对齐
3. 引入姿态一致性约束 (Orientation Alignment)
4. 实现可微分的轨迹积分
"""

from typing import List, Optional, Tuple, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

from starVLA.model.modules.dino_model.dino import get_dino_model
from starVLA.model.modules.map_model import get_map_model
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# ================= 工具函数：可微分几何变换 =================

class MetricScaleModulator(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) Layer for Metric Scale.
    让 MapAnything 的 Metric Scale Token 动态调整动作特征的幅度。
    这比显式乘以一个 float scale 优雅得多，因为它允许非线性调整。
    """
    def __init__(self, scale_dim: int, hidden_dim: int):
        super().__init__()
        self.scale_mlp = nn.Sequential(
            nn.Linear(scale_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2) # 输出 gamma (缩放) 和 beta (偏移)
        )
        
    def forward(self, x: torch.Tensor, scale_token: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, H] - Action Model 的 Latent Features
            scale_token: [B, 1, C] - MapAnything 的 Metric Token
        """
        # 计算调制参数
        scale_params = self.scale_mlp(scale_token) # [B, 1, 2*H]
        gamma, beta = scale_params.chunk(2, dim=-1) # gamma: [B, 1, H], beta: [B, 1, H]
        
        # 应用调制: y = gamma * x + beta
        # 这实际上是在告诉网络：“现在的空间尺度是这样的，请调整你的激活值”
        return x * (1 + gamma) + beta

class GeometricAdapter(nn.Module):
    """
    将高维 MapAnything 特征适配到 Action Model 的空间。
    使用卷积保留局部几何感知能力，而不是简单的 Linear。
    """
    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        # 使用 1x1 卷积降维，然后使用 Depthwise 卷积混合局部信息
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_dim, kernel_size=1),
            nn.GroupNorm(8, out_dim),
            nn.SiLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim), # 深度卷积保留几何结构
            nn.SiLU()
        )
        self.out_dim = out_dim

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: [B, C, H, W] (MapAnything 的单视或融合特征)
        Returns:
            flat_tokens: [B, H*W, out_dim]
        """
        B, C, H, W = feature_map.shape
        feat = self.adapter(feature_map) # [B, out_dim, H, W]
        
        # Flatten for Transformer: [B, out_dim, H*W] -> [B, H*W, out_dim]
        return feat.flatten(2).permute(0, 2, 1)

class StateAwareGeometricAdapter(nn.Module):
    """
    状态感知的几何适配器。
    作用：利用机器人的本体状态（State）来告诉模型“重点看地图的哪一块”。
    """
    def __init__(self, map_channels, state_dim=7, embed_dim=1024):
        super().__init__()
        
        # 1. State Encoder: 将物理状态映射到语义空间
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.SiLU(),
            nn.Linear(256, embed_dim)
        )
        
        # 2. Scale Encoder: 将 MapAnything 的尺度映射过来
        self.scale_mlp = nn.Sequential(
            nn.Linear(map_channels, 256),
            nn.SiLU(),
            nn.Linear(256, embed_dim)
        )
        
        # 3. Spatial Attention Gate
        # 输入: Map Features + (State + Scale)
        # 输出: 空间注意力权重图
        self.attn_conv = nn.Sequential(
            nn.Conv2d(map_channels + embed_dim, 512, kernel_size=1), # 融合通道
            nn.SiLU(),
            nn.Conv2d(512, 1, kernel_size=1), # 输出单通道 Attention Map
            nn.Sigmoid()
        )
        
        # 4. 最终特征映射
        self.out_proj = nn.Conv2d(map_channels, embed_dim, kernel_size=1)

    def forward(self, 
                map_features: torch.Tensor,   # [B, C, H, W]
                metric_scale: torch.Tensor,   # [B, 1, C]
                robot_state: torch.Tensor     # [B, 7]
                ) -> torch.Tensor:
        
        # print("StateAwareGeometricAdapter: map_features.shape:", map_features.shape)
        B, C, H, W = map_features.shape
        
        # A. 编码 State 和 Scale
        # robot_state: [B, 7] -> [B, D]
        state_emb = self.state_mlp(robot_state)
        # metric_scale: [B, 1, C] -> [B, C] -> [B, D]
        # scale_emb = self.scale_mlp(metric_scale.squeeze(1))
        scale_emb = self.scale_mlp(metric_scale.reshape(B, -1)) # [B, 1, C] -> [B, C] -> [B, D]
        
        # 融合物理上下文: "我现在的状态 + 环境的尺度"
        physical_context = state_emb + scale_emb # [B, D]
        
        # B. 广播上下文以匹配图像尺寸
        # [B, D, 1, 1] -> [B, D, H, W]
        context_grid = physical_context.view(B, -1, 1, 1).expand(-1, -1, H, W)
        
        # C. 计算注意力 (State-Guided Attention)
        # 拼接: [B, C, H, W] + [B, D, H, W] -> [B, C+D, H, W]
        combined = torch.cat([map_features, context_grid], dim=1)
        
        # 得到 Attention Map: [B, 1, H, W]
        # 这张图代表：根据当前手臂位置，哪些像素是重要的？
        attention_map = self.attn_conv(combined)
        
        # D. 加权特征
        weighted_features = map_features * attention_map
        
        # E. 投影并展平
        out = self.out_proj(weighted_features) # [B, D, H, W]
        return out.flatten(2).permute(0, 2, 1), attention_map # 返回 Features 和 Attention用于可视化

# ================= 主模型框架 =================

@FRAMEWORK_REGISTRY.register("QwenSuper-MapAnything")
class QwenSuperMapAnything(baseframework):
    """
    Elegant Version:
    不再显式计算几何约束。
    而是将 MapAnything 作为 "World Model"，通过 Attention 和 Modulation 
    隐式地教会 DiT 物理空间规则。
    """
    
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config
        
        # 1. VLM Interface (Planner/Observer)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        H_QWEN = self.qwen_vl_interface.model.config.hidden_size
        
        # 2. Action Model (DiT - The Controller)
        # 确保 DiT 知道输入 State 的维度是 7
        self.config.framework.action_model.state_dim = 7 
        self.config.framework.action_model.action_dim = 7
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = H_QWEN
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        
        # Dino (用于手腕相机或补充纹理信息)
        self.dino_encoder = get_dino_model(
            backbone_name=getattr(self.config.framework.dino, "dino_backbone", "dinov2_vits14")
        )
        self.dino_pro = nn.Linear(
            in_features=self.dino_encoder.num_channels, 
            out_features=H_QWEN
        )

        # 2. MapAnything (The World Grounding)
        self.map_encoder = get_map_model(self.config.framework.map_anything)
        C_MAP = self.map_encoder.info_sharing.dim

        # 3. Elegant Fusion Modules
        # 几何适配器：把 MapAnything 的特征变成 DiT 听得懂的语言
        # self.geo_adapter = GeometricAdapter(in_channels=C_MAP, out_dim=H_QWEN)

        self.geo_adapter = StateAwareGeometricAdapter(
            map_channels=C_MAP, 
            state_dim=7, 
            embed_dim=H_QWEN
        )
        
        # 尺度调制器：把 "Scale" 变成 DiT 的 "Gain"
        self.scale_modulator = MetricScaleModulator(scale_dim=C_MAP, hidden_dim=H_QWEN)

        # Hyperparams
        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
    def forward(self, examples: List[dict] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        前向传播：融合 VLM 语义与 MapAnything 物理约束
        """
        # 1. 数据对齐
        batch_images, wrist_views, instructions, state = self.align_model_input(examples)
     
        # 2. VLM & DINO Inference (Semantic & Texture)
        # last_hidden: [B, L_qwen + L_dino, H] (dtype=BF16 from autocast)
        # state: [B, 1, 7] (dtype=BF16)
        last_hidden, state = self.get_action_condition(batch_images, instructions, wrist_views, state)
        
        # 3. MapAnything Inference (Metric Geometry)
        # MapAnything 通常运行在 FP32 (或者它自己内部的混合精度)，且在 prepare 时转为了 device
        map_input = self.map_encoder.prepare_map_input(examples)
        map_output = self.map_encoder(map_input)

        spatial_feats = map_output["spatial_features"] # [B, V, C, H, W] (FP32)
        metric_scale = map_output["metric_scale"]      # [B, 1, C] (FP32)

        # print("spatial_feats.shape:", spatial_feats.shape)
        # print("metric_scale.shape:", metric_scale.shape)

        # 4. Feature Fusion (The Elegant Part)
        # A. 处理空间特征 - 假设 V=0 是主操作视角
        main_view_feat = spatial_feats[:, 0] # [B, C, H, W]

        # B. 准备 State (关键修复：转换 dtype 以匹配 Adapter)
        target_dtype = self.geo_adapter.state_mlp[0].weight.dtype # 获取 Adapter 的权重类型(通常是FP32)
        
        if state is None:
            state_for_adapter = torch.zeros((len(examples), 7), device=last_hidden.device, dtype=target_dtype)
        else:
            # 这里的 state 是 BF16，需要转回 FP32 喂给 Adapter
            state_for_adapter = state.squeeze(1) if state.dim() == 3 else state
            state_for_adapter = state_for_adapter.to(dtype=target_dtype)

        # 让 Robot State 告诉网络应该关注 Feature Map 的哪一部分
        geo_tokens, debug_attn = self.geo_adapter(
            main_view_feat, 
            metric_scale, 
            state_for_adapter 
        )
        
        # C. 拼接 Condition
        # geo_tokens 是 FP32, last_hidden 是 BF16
        # 我们需要决定在哪个精度下拼接。通常 Action Model 输入如果是 BF16，则转 geo_tokens
        geo_tokens = geo_tokens.to(dtype=last_hidden.dtype) # 转为 BF16 拼接
        
        # [B, L_total, H]
        fused_condition = torch.cat([last_hidden, geo_tokens], dim=1)
        
        # D. 应用 Metric Scale Modulation
        # Scale Modulator 的输入也需要类型对齐
        # metric_scale 是 FP32，modulator 权重是 FP32，输入 x 是 BF16
        # 为了稳定，我们在 modulator 内部处理，或者先把 modulator 转为 BF16
        # 这里最稳妥的是把 metric_scale 转为 BF16 (匹配 x)，并确保 modulator 能够处理混合精度
        # 或者在调用前把 metric_scale 转为 modulator 权重的类型
        metric_scale_for_mod = metric_scale.to(dtype=self.scale_modulator.scale_mlp[0].weight.dtype)
        
        # 如果 modulator 是 FP32，输入 x (BF16) 会自动强转 FP32 计算，然后输出 FP32
        # 我们最后再转回 BF16
        fused_condition = self.scale_modulator(fused_condition.float(), metric_scale_for_mod).to(dtype=last_hidden.dtype)

        # 5. Action Prediction Loop
        # --- 2. 动作预测 (Action Prediction) ---
        with torch.autocast("cuda", dtype=torch.float32):
            # get action labels
            actions = [example["action"] for example in examples]  # List of [T_full, action_dim]
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            # repeate for efficient training
            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            fused_condition_repeated = fused_condition.repeat(repeated_diffusion_steps, 1, 1)
            state_repeated = state.repeat(repeated_diffusion_steps, 1, 1) if state is not None else None
            
            action_loss = self.action_model(fused_condition_repeated, actions_target_repeated, state_repeated)  # (B, chunk_len, action_dim)

        return {
            "action_loss": action_loss,
            "debug_scale_magnitude": metric_scale.mean().item()
        }

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        优雅的推理：不需要任何手动几何检查。
        模型应该因为训练时的 Scale Modulation 而自动输出符合物理尺度的动作。
        """
        batch_images, wrist_views, instructions, state = self.align_model_input(examples)
        # 1. Condition Generation
        # last_hidden (BF16), state (BF16)
        last_hidden, state = self.get_action_condition(batch_images, instructions, wrist_views, state)
        
        map_input = self.map_encoder.prepare_map_input(examples)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            map_output = self.map_encoder(map_input)

        metric_scale = map_output["metric_scale"] # FP32
        
        # 修复: dtype 转换
        target_dtype = self.geo_adapter.state_mlp[0].weight.dtype
        if state is not None:
            state_for_adapter = state.squeeze(1) if state.dim() == 3 else state
            state_for_adapter = state_for_adapter.to(dtype=target_dtype)
        else:
            state_for_adapter = torch.zeros((len(examples), 7), device=last_hidden.device, dtype=target_dtype)

        geo_tokens, _ = self.geo_adapter(
            map_output["spatial_features"][:, 0].to(dtype=target_dtype),
            metric_scale.to(dtype=target_dtype),
            state_for_adapter
        )
        geo_tokens = geo_tokens.to(dtype=last_hidden.dtype)

        # 2. Fusion & Modulation
        fused_condition = torch.cat([last_hidden, geo_tokens], dim=1)
        
        metric_scale_for_mod = metric_scale.to(dtype=self.scale_modulator.scale_mlp[0].weight.dtype)
        fused_condition = self.scale_modulator(fused_condition.float(), metric_scale_for_mod).to(dtype=last_hidden.dtype)
        
        # 3. Action Generation
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(fused_condition, state)  # (B, chunk_len, action_dim)
        normalized_actions = pred_actions.detach().cpu().numpy()
        
        return {
            "normalized_actions": normalized_actions
        }
    
    def align_model_input(self, examples: List[dict]):

        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        wrist_views = [to_pil_preserve(example["wrist_views"]) for example in examples] if "wrist_views" in examples[0] else None #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        states = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        # print(states)
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", [224,224])
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        if train_obs_image_size and wrist_views is not None:
            wrist_views = resize_images(wrist_views, target_size=train_obs_image_size)
            
        return batch_images, wrist_views, instructions, states
    
    def get_action_condition(self, batch_images, instructions, wrist_views=None, state=None):
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            connect_layer_index = self.config.framework.action_model.get("connect_layer_index", -1)
            last_hidden = qwenvl_outputs.hidden_states[connect_layer_index]   # [B, L, H]

            # Step 2: DINO Forward
            if wrist_views == None:
               wrist_views = batch_images
            image_tensors = self.dino_encoder.prepare_dino_input(wrist_views)  #
            B = len(batch_images)
            dino_features = self.dino_encoder(image_tensors)  # DINO output is [B*num_view, token, dim]
            dino_encoded_features = dino_features.reshape(B, -1, dino_features.shape[-1])  # [B, num_view * token, dim]
            dino_encoded_features = self.dino_pro(dino_encoded_features)  # [B, num_view * token, hidden_size]

            # Step 3: Feature Concatenation
            last_hidden = torch.cat(
                [last_hidden, dino_encoded_features], dim=1
            )

        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        
        return last_hidden, state
    
if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/supervla_cotrain_oxe.yaml")
    args, clipargs = parser.parse_known_args()
    
    cfg = OmegaConf.load(args.config_yaml)
    # cfg.framework.qwenvl.base_vlm = "/public/home/vlabadmin/dataset/Qwen3-VL-4B-Instruct"
    cfg.framework.qwenvl.base_vlm = "/data/models/Qwen3-VL-4B-Instruct"
    
    model = QwenSuperMapAnything(cfg)
    print(model)
    print(f"模型总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    # 测试用假数据
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # three views
        # "wrist_views": [image, image],
        "lang": "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float32),
    }
    
    sample2 = sample.copy()
    sample2["lang"] = "Move the red cup from the table to the kitchen counter next to the sink."
    batch  = [sample, sample2]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    print("Forward Output Keys:", forward_output.keys(), forward_output)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss}")

    # test predict action
    predict_output = model.predict_action([sample]) #, state=[batch[0]["state"]]
    print("Predict Output Keys:", predict_output.keys(), predict_output)
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")