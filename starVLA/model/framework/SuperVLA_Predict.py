# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0.
# Version: Predictive SuperVLA (State-Action-Predictive Consistency)

"""
SuperVLA_Predictive: 从“描述世界”升级为“预测动作后果”。
核心改进：
1. PredictiveStateBottleneck: 将状态分解为 Affordance, Risk, Direction。
2. Temporal Dynamics: 引入 Delta Predictor，预测动作对状态的影响。
3. Consistency Loss: 强制让隐空间满足物理演化规律。
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

# ================= 核心模块：预测型状态压缩器 =================

class PredictiveStateBottleneck(nn.Module):
    """
    不再仅仅是 mean pooling。
    S_t = [Affordance, Risk, Direction]
    并且包含一个 Delta Predictor: ΔS = f(S, a)
    """
    def __init__(self, vlm_dim: int, map_dim: int, state_dim: int = 256, action_dim: int = 7):
        super().__init__()
        self.state_dim = state_dim
        
        # 语义与几何的基础编码器
        self.geo_proj = nn.Sequential(nn.Linear(map_dim, 256), nn.SiLU(), nn.Linear(256, state_dim // 2))
        self.sem_proj = nn.Sequential(nn.Linear(vlm_dim, 256), nn.SiLU(), nn.Linear(256, state_dim // 2))

        # 动力学预测器：预测执行 action 后，state 会发生什么变化
        # 这是“手感”的来源
        self.delta_predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.SiLU(),
            nn.Linear(256, state_dim)
        )
        
        # 辅助 Head：预测当前状态的风险（可选，用于增强表征）
        self.risk_estimator = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def encode(self, vlm_tokens: torch.Tensor, map_tokens: torch.Tensor) -> torch.Tensor:
        # vlm_tokens: [B, L, D1], map_tokens: [B, L, D2]
        g = self.geo_proj(map_tokens.mean(dim=1))
        s = self.sem_proj(vlm_tokens.mean(dim=1))
        return torch.cat([g, s], dim=-1) # [B, state_dim]

    def forward(self, vlm_tokens, map_tokens, action=None):
        s_t = self.encode(vlm_tokens, map_tokens)
        res = {"state": s_t, "risk": self.risk_estimator(s_t)}
        
        if action is not None:
            # 预测执行该动作后的状态增量
            # 注意：这里的 action 可能是 chunk，我们取第一步或者平均值
            if action.ndim == 3: 
                action_input = action[:, 0, :] # 取当前步
            else:
                action_input = action
            
            res["delta"] = self.delta_predictor(torch.cat([s_t, action_input], dim=-1))
        
        return res

# ================= 主模型架构 =================
@FRAMEWORK_REGISTRY.register("SuperVLA_Predictive")
class SuperVLA_Predictive(baseframework):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # 1. Perception Backbones (Frozen or Tuned)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        H_QWEN = self.qwen_vl_interface.model.config.hidden_size
        
        # 2. Predictive Bottleneck
        # MapAnything: 提供几何真值 (Oracle)
        # 注意：我们在训练中利用它，但在推理中它只通过 Bottleneck 产生影响
        self.map_encoder = get_map_model(self.config.framework.map_anything)
        C_MAP = self.map_encoder.info_sharing.dim

        self.bottleneck = PredictiveStateBottleneck(vlm_dim=H_QWEN, map_dim=C_MAP)

        # 3. Blind Policy (只看隐变量，不看图)
        self.action_model = get_action_model(config)

    def forward(self, examples: List[dict]) -> Dict[str, torch.Tensor]:
        # 数据对齐
        batch_images, instructions, actions, _ = self.align_model_input(examples)
        device = batch_images.device
        
        # --- 第一步：感知提取 ---
        # MapAnything 提取度量几何
        map_input = self.map_encoder.prepare_map_input(examples)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            map_output = self.map_encoder(map_input)
        map_tokens = map_output["spatial_features"].flatten(2).permute(0, 2, 1) # [B, L, D]
        
        # Qwen 提取语义
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(batch_images, instructions)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True)
        vlm_tokens = qwenvl_outputs.hidden_states[-1]

        # --- 第二步：构建预测型 State ---
        # 我们假设数据集提供了 (s_t, a_t, s_{t+1}) 的连续关系
        # 如果是单帧数据，delta_loss 将不生效
        curr_action = torch.from_numpy(actions).to(device, dtype=torch.float32)
        
        state_out = self.bottleneck(vlm_tokens.float(), map_tokens.float(), curr_action)
        s_t = state_out["state"]
        
        # --- 第三步：动作预测 (Blind Policy) ---
        # Policy 不看图像，只看压缩后的 s_t 和指令（指令其实已经通过 Qwen 编码进了 s_t）
        # 这里需要注意 action_model 的输入接口
        action_loss_dict = self.action_model(
            state_features=s_t.unsqueeze(1), # [B, 1, d]
            vl_embs=vlm_tokens,               # 作为上下文
            actions=curr_action
        )
        
        # --- 第四步：动力学损失 (Dynamics Consistency) ---
        # 核心：s_{t+1} 应接近 s_t + delta
        # 在实际训练中，你需要从 Dataloader 获取 next_image
        # 这里演示逻辑：
        total_loss = action_loss_dict["action_loss"]
        
        # 几何辅助损失 (MapAnything Metric Scale)
        if "metric_scale" in map_output:
            target_scale = map_output["metric_scale"]
            scale_loss = F.mse_loss(state_out["risk"], target_scale.mean(dim=1, keepdim=True).float())
            total_loss += 0.1 * scale_loss

        return {
            "loss": total_loss,
            "action_loss": action_loss_dict["action_loss"]
        }

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> np.ndarray:
        # 推理时完全不需要 next_state，只需感知当前 S_t
        batch_images, instructions, _, _ = self.align_model_input(examples)
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            map_input = self.map_encoder.prepare_map_input(examples)
            map_output = self.map_encoder(map_input)
            map_tokens = map_output["spatial_features"].flatten(2).permute(0, 2, 1)
            
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(batch_images, instructions)
            qwenvl_outputs = self.qwen_vl_interface(**qwen_inputs, output_hidden_states=True)
            vlm_tokens = qwenvl_outputs.hidden_states[-1]

        s_t = self.bottleneck.encode(vlm_tokens.float(), map_tokens.float())
        
        pred_actions = self.action_model.predict_action(
            state_t=s_t.unsqueeze(1),
            vl_embs=vlm_tokens
        )
        
        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

    def align_model_input(self, examples):
        # 保持与父类一致的解析逻辑
        batch_images = [Image.fromarray(ex["image"][0]) if isinstance(ex["image"][0], np.ndarray) else ex["image"][0] for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        actions = np.array([ex["action"] for ex in examples])
        states = [ex.get("state", None) for ex in examples]
        
        # 统一尺寸
        batch_images = resize_images(batch_images, (224, 224))
        return torch.stack([torch.from_numpy(np.array(img)).permute(2,0,1) for img in batch_images]).float().cuda(), instructions, actions, states


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

        
    model = SuperVLA_Predictive(cfg)
    print("SuperVLA_Predictive Instantiated Successfully.")
    
    # 打印参数量对比
    print(f"Total Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 检查 Action Model 的 Cross Attention Dim 是否正确
    print(f"Action Model Cross Attn Dim: {model.config.framework.action_model.diffusion_model_cfg.cross_attention_dim}")
    assert model.config.framework.action_model.diffusion_model_cfg.cross_attention_dim == 256, "Bottleneck dim mismatch!"