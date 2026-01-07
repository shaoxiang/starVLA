# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0.
# Implemented by Gemini & User based on "Action Follows State" Hypothesis.

"""
SuperVLA_Minimal: A "Glass Box" Architecture.
核心理念：
1. State Bottleneck: 强制所有感知信息压缩为低维结构化状态 S_t。
2. Blind Policy: 策略网络完全不看图像 Token，只看 S_t。
3. Metric Grounding: 利用 MapAnything 的物理尺度监督 S_t 的构建，而非作为输入。
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

# ================= 核心模块：结构化状态压缩器 =================

class StructuredStateBottleneck(nn.Module):
    """
    负责将高维的 VLM 语义特征和 MapAnything 几何特征，
    压缩为显式的、低维的物理状态 S_t。
    
    Structure of S_t (Latent but Grounded):
    [ Scene_Context (Geometry) | Task_Intent (Semantic) | Proprioception (Robot) ]
    """
    def __init__(self, vlm_dim: int, map_dim: int, state_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        
        # 1. 几何压缩 (Geometry Compressor)
        # 将 MapAnything 的 patch tokens 压缩为全局几何上下文
        self.geo_proj = nn.Sequential(
            nn.Linear(map_dim, 256),
            nn.SiLU(),
            nn.Linear(256, state_dim // 2) # 分配一半维度给几何
        )
        
        # 2. 语义压缩 (Semantic Compressor)
        # 将 Qwen 的 visual tokens 压缩为任务意图
        self.sem_proj = nn.Sequential(
            nn.Linear(vlm_dim, 256),
            nn.SiLU(),
            nn.Linear(256, state_dim // 2) # 分配一半维度给语义
        )
        
        # 3. 物理尺度预测头 (Metric Scale Head)
        # 用于 Auxiliary Loss：从状态中回归出真实的物理尺度
        self.scale_pred_head = nn.Linear(state_dim, 1)

    def forward(self, vlm_tokens: torch.Tensor, map_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vlm_tokens: [B, L_v, D_v] (Qwen features)
            map_tokens: [B, N_m, D_m] (MapAnything spatial features)
        Returns:
            state_t: [B, 1, state_dim] (The Bottleneck)
            pred_scale: [B, 1] (Predicted metric scale for loss)
        """
        # A. 简单的 Attention Pooling 或 Mean Pooling
        # 这里使用 Mean Pooling 以保持最简 Minimal 设计，避免引入额外参数导致过拟合
        global_geo = map_tokens.mean(dim=1) # [B, D_m]
        global_sem = vlm_tokens.mean(dim=1) # [B, D_v]
        
        # B. 投影到瓶颈空间
        s_geo = self.geo_proj(global_geo) # [B, S/2]
        s_sem = self.sem_proj(global_sem) # [B, S/2]
        
        # C. 拼接构成 S_t
        state_t = torch.cat([s_geo, s_sem], dim=-1) # [B, S]
        
        # D. 预测尺度 (用于自我监督)
        pred_scale = self.scale_pred_head(state_t)
        
        return state_t.unsqueeze(1), pred_scale

# ================= 主模型框架 =================

@FRAMEWORK_REGISTRY.register("SuperVLA_Minimal")
class SuperVLA_Minimal(baseframework):
    """
    The Minimal "Glass Box" VLA.
    Action Head is BLIND to visual tokens. It only sees the State Bottleneck.
    MapAnything acts as a Teacher (Grounding Source), not a Feature Provider.
    """
    
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config
        
        # 1. Perception (Frozen/Slow Systems)
        # Qwen2-VL: 提供语义理解
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        H_QWEN = self.qwen_vl_interface.model.config.hidden_size
        
        # MapAnything: 提供几何真值 (Oracle)
        # 注意：我们在训练中利用它，但在推理中它只通过 Bottleneck 产生影响
        self.map_encoder = get_map_model(self.config.framework.map_anything)
        C_MAP = self.map_encoder.info_sharing.dim
        
        # 2. State Bottleneck (The Core Innovation)
        # 我们定义一个 256 维的瓶颈状态，这比原始的 Token 序列小得多
        self.state_dim = 256
        self.bottleneck = StructuredStateBottleneck(vlm_dim=H_QWEN, map_dim=C_MAP, state_dim=self.state_dim)
        
        # 3. Action Model (The Blind Policy)
        # === 关键修改 ===
        # 强制修改 Action Model 的输入维度配置
        # 让 DiT 的 Cross-Attention 只接受 state_dim 大小的输入，而不是 Qwen 的 hidden_size
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.state_dim
        # 显式状态维度 (x,y,z,r,p,y,g)
        self.config.framework.action_model.state_dim = 7 
        self.config.framework.action_model.action_dim = 7
        
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        
        # 4. 辅助模块
        # DINO 用于补充纹理细节 (Optional, integrated into Qwen path usually, but keeping simple here)
        # 在这个 Minimal 版本中，我们暂时移除 DINO 的直接输入，依靠 Qwen 足够强大
        # 以确保 "Blind Policy" 的纯粹性
        
        # Hyperparams
        self.future_action_window_size = config.framework.action_model.future_action_window_size
        
    def forward(self, examples: List[dict] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Training Pass:
        Image -> Qwen/Map -> Bottleneck -> (Action Loss + Grounding Loss)
        """
        # 1. 数据准备
        batch_images, instructions, actions, robot_states = self.align_model_input(examples)
        
        # 2. Perception Pass (Feature Extraction)
        # MapAnything (Teacher) - 获取几何特征和真实的 Metric Scale
        # 我们不需要它的梯度，它只是用来"锚定"我们的状态
        with torch.no_grad(): 
            map_input = self.map_encoder.prepare_map_input(examples)
            map_output = self.map_encoder(map_input)
            spatial_feats = map_output["spatial_features"][:, 0] # [B, C, H, W]
            gt_metric_scale = map_output["metric_scale"]         # [B, 1, C] -> 简化为一个标量
            # 简化 metric scale 为全局平均尺度，用于监督
            gt_scale_scalar = gt_metric_scale.mean(dim=-1).float() # [B, 1]

            # Flatten map features for compressor
            map_tokens = spatial_feats.flatten(2).permute(0, 2, 1) # [B, N, C]

        # Qwen-VL (Semantic)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs, output_hidden_states=True, return_dict=True
            )
            vlm_tokens = qwenvl_outputs.hidden_states[-1] # [B, L, H]

        # 3. State Compression (The Bottleneck)
        # 必须转为 float32 进行稳定的状态计算
        with torch.autocast("cuda", dtype=torch.float32):
            # 输入: VLM tokens (BF16->FP32), Map tokens (FP32)
            # 输出: State S_t, Predicted Scale
            state_t, pred_scale = self.bottleneck(vlm_tokens.float(), map_tokens)
            
            # 融合 Proprioception (Robot State) 到 State S_t 中
            # 这里我们简单地将 robot_state 作为 Condition 给 Action Model，
            # 或者你可以选择将其 concat 到 state_t 中。
            # 为了保持 Action Model 接口一致，我们通过 standard argument 传入 robot_state
            
        # 4. Blind Policy Execution
        # Action Model 只能看到 state_t [B, 1, 256]，看不到 vlm_tokens！
        with torch.autocast("cuda", dtype=torch.float32):
            actions_tensor = torch.tensor(np.array(actions), device=state_t.device).float()
            # 这样 state_encoder 输出的也会是 [B, 1, Hidden]，可以作为 token 拼接
            robot_state_tensor = torch.tensor(np.array(robot_states), device=state_t.device).float()
            if robot_state_tensor.dim() == 2:
                robot_state_tensor = robot_state_tensor.unsqueeze(1)

            # Action Loss (Flow Matching)
            # 注意：condition 是 state_t
            # 我们需要重复 state_t 以匹配 action chunk (如果 ActionHead 内部没处理)
            # GR00T Action Head 通常支持 broadcasting condition
            action_loss = self.action_model(state_t, actions_tensor, robot_state_tensor)

        # 5. Auxiliary Loss: Grounding (锚定损失)
        # 强迫 S_t 包含物理尺度信息
        # 如果 S_t 预测的尺度错了，说明它没理解几何
        scale_loss = F.mse_loss(pred_scale, gt_scale_scalar)
        
        total_loss = action_loss + 0.2 * scale_loss

        return {
            "loss": total_loss,
            "action_loss": action_loss,
            "scale_loss": scale_loss,
            "debug_gt_scale": gt_scale_scalar.mean().item(),
            "debug_pred_scale": pred_scale.mean().item()
        }
    
    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        推理流程
        """
        batch_images, instructions, _, robot_states = self.align_model_input(examples)
        
        # 1. Perception
        # 修复：使用与模型权重一致的自动精度转换 (bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            map_input = self.map_encoder.prepare_map_input(examples)
            map_output = self.map_encoder(map_input)
            
        spatial_feats = map_output["spatial_features"][:, 0]
        map_tokens = spatial_feats.flatten(2).permute(0, 2, 1)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True)
            vlm_tokens = qwenvl_outputs.hidden_states[-1]
            
        # 2. State Compression
        # Bottleneck 通常在 float32 下更稳定
        with torch.autocast("cuda", dtype=torch.float32):
            state_t, _ = self.bottleneck(vlm_tokens.float(), map_tokens.float())
        
        # 3. Blind Policy
        robot_state_tensor = torch.from_numpy(np.array(robot_states)).to(state_t.device, dtype=torch.float32)
        if robot_state_tensor.dim() == 2:
            robot_state_tensor = robot_state_tensor.unsqueeze(1)
        
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(state_t, robot_state_tensor)
            
        return {
            "normalized_actions": pred_actions.detach().cpu().numpy()
        }
    
    def align_model_input(self, examples: List[dict]):
        # 标准的数据对齐逻辑，与 QwenSuper 保持一致
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        
        actions = None
        if "action" in examples[0]:
            actions = [example["action"] for example in examples]

        states = None
        if "state" in examples[0]:
            states = [example["state"] for example in examples]
        
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", [224,224])
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
            
        return batch_images, instructions, actions, states

if __name__ == "__main__":
    # 简单的测试代码块，确保能够运行
    from omegaconf import OmegaConf
    import argparse
    
    # 模拟 Config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/supervla_cotrain_oxe.yaml")
    args, _ = parser.parse_known_args()
    
    # 注意：这里假设你有相应的 config 文件
    # 如果没有，你需要手动构建一个 dummy config
    try:
        cfg = OmegaConf.load(args.config_yaml)
        # 强制设置 MapAnything 路径等
        cfg.framework.qwenvl.base_vlm = "/data/models/Qwen3-VL-4B-Instruct"
    except:
        print("Warning: Config file not found, skipping instantiation test.")
        exit()

        
    model = SuperVLA_Minimal(cfg)
    print("SuperVLA_Minimal Instantiated Successfully.")
    
    # 打印参数量对比
    print(f"Total Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 检查 Action Model 的 Cross Attention Dim 是否正确
    print(f"Action Model Cross Attn Dim: {model.config.framework.action_model.diffusion_model_cfg.cross_attention_dim}")
    assert model.config.framework.action_model.diffusion_model_cfg.cross_attention_dim == 256, "Bottleneck dim mismatch!"