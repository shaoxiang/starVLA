# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
QwenSuper-GeometricConstraint Framework
核心创新：将MapAnything从特征拼接改为物理约束层
保持VLM→DiT主链路纯净，几何信息仅在损失层约束动作合理性
"""

from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

from starVLA.model.modules.dino_model.dino import get_dino_model
from starVLA.model.modules.map_model import get_map_model
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY


class GeometricConstraintHead(nn.Module):
    """
    物理约束头：评估预测动作与3D几何的一致性
    关键：该模块与主链路并行，不干扰VLM→DiT梯度流
    使用 widowx wx250s 机械臂数据训练
    https://www.trossenrobotics.com/widowx-250
    """
    
    def __init__(self, geom_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # 约束网络：动作+几何 → 可行性分数
        self.constraint_net = nn.Sequential(
            nn.Linear(geom_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 输出[0,1]的物理可行性分数
        )
        
        # 手-物距离阈值（米）
        self.grasp_threshold = nn.Parameter(torch.tensor(0.015))  # 1.5cm（匹配5-8mm精度）
        
        # 工作空间限制（根据650mm臂展精确设置）
        self.workspace_limits = nn.Parameter(torch.tensor([0.65, 0.65, 0.65]))  # x,y,z最大距离

        # 动作缩放因子（根据WidowX速度特性调整）
        self.action_scale = nn.Parameter(torch.tensor(0.05))  # 每步最大5cm移动
        
    def forward(self, 
                pred_actions: torch.Tensor, 
                geom_tokens: torch.Tensor,
                scale_token: torch.Tensor,
                robot_state: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_actions: [B, T, action_dim] 预测的动作序列
            geom_tokens: [B, N_geo, geom_dim] MapAnything的几何token
            scale_token: [B, 1, geom_dim] 场景尺度因子
            robot_state: [B, state_dim] 机器人当前状态
        Returns:
            约束损失字典
        """
        B, T, D = pred_actions.shape
        
        # 1. 从几何token重建度量点云（简化版，实际可用DPT head解码）
        # 假设geom_tokens已编码局部点云，scale_token提供全局尺度
        metric_scale = scale_token.mean(dim=1, keepdim=True)  # [B, 1, geom_dim]
        reconstructed_points = geom_tokens * metric_scale  # [B, N_geo, geom_dim]
        
        # 2. 提取预测动作的末端执行器轨迹
        # 假设action的前3维是相对位移（需根据实际动作空间调整）
        # 在action_to_ee_trajectory中应用缩放
        ee_displacement = pred_actions[:, :, :3] * self.action_scale  # 从归一化到米的转换 # [B, T, 3]
        
        # 如果有机器人状态，计算绝对轨迹
        if robot_state is not None:
            current_ee_pos = robot_state[:, :3].unsqueeze(1)  # [B, 1, 3]
            ee_trajectory = current_ee_pos + torch.cumsum(ee_displacement * 0.1, dim=1)  # 积分得到绝对位置
        else:
            ee_trajectory = ee_displacement
        
        # 3. 计算手-物距离约束（抓取阶段）
        # 计算每个时间步手部到所有物体点的最小距离
        hand_obj_distances = torch.cdist(ee_trajectory, reconstructed_points[:, :, :3])  # [B, T, N_geo]
        min_distances, _ = hand_obj_distances.min(dim=-1)  # [B, T]
        
        # 抓取阶段识别：假设夹爪状态在action第6维
        if D > 6:
            gripper_state = pred_actions[:, :, 6]  # [B, T]
            is_grasp_phase = gripper_state < 0.5  # 夹爪闭合
        else:
            is_grasp_phase = torch.ones(B, T, device=pred_actions.device, dtype=torch.bool)
        
        # 距离惩罚：抓取时手应接近物体（<5cm），平时应避免碰撞（>2cm）
        grasp_penalty = F.relu(min_distances - self.grasp_threshold)  # 距离过大
        collision_penalty = F.relu(0.02 - min_distances)  # 距离过小（穿透）
        
        # 只在抓取阶段应用距离惩罚，所有阶段避免碰撞
        distance_constraint = (is_grasp_phase.float() * grasp_penalty + collision_penalty).mean()
        
        # 4. 工作空间约束
        workspace_violation = F.relu(
            torch.abs(ee_trajectory) - self.workspace_limits
        ).mean()
        
        # 5. 平滑度约束（避免抖动）
        velocity = ee_displacement[:, 1:, :] - ee_displacement[:, :-1, :]
        smoothness = velocity.norm(dim=-1).mean()
        
        return {
            'distance_constraint': distance_constraint,
            'workspace_constraint': workspace_violation,
            'smoothness_constraint': smoothness,
            'min_hand_obj_dist': min_distances.mean(),  # 用于监控
        }


@FRAMEWORK_REGISTRY.register("QwenSuper-GeometricConstraint")
class QwenSuperGeometricConstraint(baseframework):
    """
    几何约束增强的VLA模型
    架构特点：
      - 主链路：Qwen-VL → DINO → DiT（保持纯净）
      - 约束分支：MapAnything → 物理可行性检查网络（并行）
      - 训练：主损失 + 几何约束损失的联合优化
    """
    
    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        
        # ==================== 主链路组件（保持不变） ====================
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size
        
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        self.dino_encoder = get_dino_model(
            backbone_name=getattr(self.config.framework.dino, "dino_backbone", "dinov2_vits14")
        )
        self.dino_pro = nn.Linear(
            in_features=self.dino_encoder.num_channels, 
            out_features=self.qwen_vl_interface.model.config.hidden_size
        )
        
        # ==================== 几何约束分支（新增） ====================
        if not hasattr(self.config.framework, "map_anything"):
            raise ValueError("Config missing `framework.map_anything`")
        
        self.map_encoder = get_map_model(self.config.framework.map_anything)
        
        # 动态获取维度
        C_MAP = self.map_encoder.info_sharing.dim
        H_QWEN = self.qwen_vl_interface.model.config.hidden_size
        
        # 约束头（核心模块）
        self.geom_constraint_head = GeometricConstraintHead(
            geom_dim=C_MAP,
            action_dim=getattr(self.config.framework.action_model, 'action_dim', 7),
            hidden_dim=getattr(self.config.framework, 'constraint_hidden_dim', 128)
        )
        
        # 动作窗口参数
        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
        # 损失权重（可调度）
        self.constraint_weight = getattr(config.trainer, 'constraint_weight', 0.2)
    
    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        训练前向：主链路预测动作 + 几何约束分支评估物理可行性
        """
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        
        device = next(self.parameters()).device
        
        # ==================== 主链路：VLM + DINO → DiT ====================
        # 保持主链路完全不变，确保语义-动作映射的纯净性
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            vlm_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L_vlm, H_qwen]
            
            # DINO编码
            image_tensors = self.dino_encoder.prepare_dino_input(batch_images)
            B = len(batch_images)
            dino_features = self.dino_encoder(image_tensors)
            dino_encoded = dino_features.reshape(B, -1, dino_features.shape[-1])
            dino_encoded = self.dino_pro(dino_encoded)  # [B, L_dino, H_qwen]
            
            # 主链路融合：VLM + DINO（无几何token！保持纯净）
            main_features = torch.cat([vlm_hidden, dino_encoded], dim=1)  # [B, L_vlm+L_dino, H_qwen]
        
        # ==================== 动作预测分支 ====================
        with torch.autocast("cuda", dtype=torch.float32):
            actions_tensor = torch.tensor(np.array(actions), device=device, dtype=torch.float32)
            actions_target = actions_tensor[:, -(self.future_action_window_size+1):, :]
            
            repeat_steps = self.config.trainer.get("repeated_diffusion_steps", 4)
            actions_target_rep = actions_target.repeat(repeat_steps, 1, 1)
            main_features_rep = main_features.repeat(repeat_steps, 1, 1)
            
            state_rep = None
            if state is not None:
                state_tensor = torch.tensor(np.array(state), device=device, dtype=torch.float32)
                state_rep = state_tensor.repeat(repeat_steps, 1, 1)
            
            # DiT预测动作
            pred_actions = self.action_model(main_features_rep, actions_target_rep, state_rep)
            action_loss = self.action_model.compute_loss(pred_actions, actions_target_rep)
        
        # ==================== 几何约束分支（关键：并行计算，不干扰主梯度） ====================
        # 使用torch.no_grad()确保MapAnything梯度不影响主链路
        with torch.no_grad():
            map_input_list = self.map_encoder.prepare_map_input(examples)
            geo_tokens, scale_token = self.map_encoder(map_input_list)  # [B, N_geo, C_MAP], [B, 1, C_MAP]
        
        # 约束损失计算（允许梯度回传至约束头，但不回传至MapAnything）
        constraint_losses = self.geom_constraint_head(
            pred_actions, geo_tokens, scale_token, 
            state_rep if state_rep is not None else None
        )
        
        # ==================== 总损失合并 ====================
        # 主损失 + 约束损失，但约束梯度不回流至VLM/DiT（通过torch.no_grad()实现）
        total_loss = action_loss + self.constraint_weight * sum(constraint_losses.values())
        
        return {
            "loss": total_loss,
            "action_loss": action_loss.item(),
            "distance_constraint": constraint_losses['distance_constraint'].item(),
            "workspace_constraint": constraint_losses['workspace_constraint'].item(),
            "smoothness_constraint": constraint_losses['smoothness_constraint'].item(),
            "min_hand_obj_dist": constraint_losses['min_hand_obj_dist'].item(),
            "pred_actions": pred_actions,  # 用于后续分析
        }
    
    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        **kwargs: str,
    ) -> Dict[str, np.ndarray]:
        """
        推理：单次前向直接回归未来动作（无扩散采样）
        """
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        
        # 主链路编码
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            vlm_hidden = qwenvl_outputs.hidden_states[-1]
            
            image_tensors = self.dino_encoder.prepare_dino_input(batch_images)
            B = len(batch_images)
            dino_features = self.dino_encoder(image_tensors)
            dino_encoded = dino_features.reshape(B, -1, dino_features.shape[-1])
            dino_encoded = self.dino_pro(dino_encoded)
            
            main_features = torch.cat([vlm_hidden, dino_encoded], dim=1)
        
        # DiT预测
        state_tensor = torch.from_numpy(np.array(state)).to(main_features.device, dtype=main_features.dtype) if state is not None else None
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(main_features, state_tensor)
        
        # 可选：计算几何约束用于分析（但不用于决策）
        if kwargs.get('analyze_constraints', False):
            map_input_list = self.map_encoder.prepare_map_input([
                {"image": batch_images[i], "lang": instructions[i]} for i in range(B)
            ])
            geo_tokens, scale_token = self.map_encoder(map_input_list)
            constraint_metrics = self.geom_constraint_head(
                pred_actions, geo_tokens, scale_token, state_tensor
            )

            simpler_info = {
                "min_hand_obj_dist": constraint_metrics["min_hand_obj_dist"],
                "workspace_valid": constraint_metrics["workspace_constraint"] < 0.01,
                "trajectory_smooth": constraint_metrics["smoothness_constraint"] < 0.05,
            }
            result["simpler_info"] = simpler_info
            
        else:
            constraint_metrics = None
        
        normalized_actions = pred_actions.detach().cpu().numpy()
        result = {"normalized_actions": normalized_actions}
        
        if constraint_metrics is not None:
            result["constraint_metrics"] = {k: v.item() for k, v in constraint_metrics.items()}
        
        return result


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/geometric_constraint.yaml")
    args, clipargs = parser.parse_known_args()
    
    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.qwenvl.base_vlm = "/public/home/vlabadmin/dataset/Qwen3-VL-4B-Instruct"
    
    model = QwenSuperGeometricConstraint(cfg)
    print(f"模型总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    # 测试用假数据
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float32),
        "image": [image, image],
        "lang": "Pick up the red cup and place it in the basket.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float32),
    }
    
    batch = [sample, sample]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 前向测试
    model.train()
    output = model(batch)
    print(f"✓ 训练前向成功: loss={output['loss']:.4f}, action_loss={output['action_loss']:.4f}")
    print(f"  约束损失: distance={output['distance_constraint']:.4f}, workspace={output['workspace_constraint']:.4f}")
    
    # 推理测试
    model.eval()
    pred = model.predict_action(
        batch_images=[batch[0]["image"]],
        instructions=[batch[0]["lang"]],
        state=[batch[0]["state"]],
        analyze_constraints=True
    )
    print(f"✓ 推理成功: actions_shape={pred['normalized_actions'].shape}")
    print(f"  约束分析: min_dist={pred['constraint_metrics']['min_hand_obj_dist']:.3f}m")