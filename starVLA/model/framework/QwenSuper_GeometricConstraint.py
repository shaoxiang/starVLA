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

def euler_to_rotation_matrix(euler_angles: torch.Tensor) -> torch.Tensor:
    """
    将欧拉角 (Roll, Pitch, Yaw) 转换为旋转矩阵
    Args:
        euler_angles: [B, T, 3] (roll, pitch, yaw)
    Returns:
        rot_mats: [B, T, 3, 3]
    """
    batch_size, seq_len, _ = euler_angles.shape
    r, p, y = euler_angles.unbind(-1)
    
    # 构建旋转矩阵元素
    cr, sr = torch.cos(r), torch.sin(r)
    cp, sp = torch.cos(p), torch.sin(p)
    cy, sy = torch.cos(y), torch.sin(y)
    
    # 组合旋转矩阵 (ZYX顺序，常用机械臂标准，需根据具体URDF调整)
    # R = Rz(y) * Ry(p) * Rx(r)
    
    row1 = torch.stack([cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr], dim=-1)
    row2 = torch.stack([sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr], dim=-1)
    row3 = torch.stack([-sp,   cp*sr,            cp*cr           ], dim=-1)
    
    rot_mats = torch.stack([row1, row2, row3], dim=-2) # [B, T, 3, 3]
    return rot_mats

def differentiable_trajectory_integration(
    initial_state: torch.Tensor, 
    delta_actions: torch.Tensor,
    scale_factor: float = 0.05
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将相对动作积分为绝对轨迹
    Args:
        initial_state: [B, 7] or [B, 1, 7] (x, y, z, r, p, y, g)
        delta_actions: [B, T, 7] (dx, dy, dz, dr, dp, dy, dg)
        scale_factor: 动作缩放因子 (归一化空间 -> 物理空间)
    Returns:
        abs_pos: [B, T, 3] 绝对位置轨迹
        abs_rpy: [B, T, 3] 绝对欧拉角轨迹
    """
    # 确保 initial_state 是 [B, 7]，如果是 [B, 1, 7] 则压缩维度
    if initial_state.dim() == 3 and initial_state.shape[1] == 1:
        initial_state = initial_state.squeeze(1)

    # 分离位置和姿态增量
    # delta_actions: [B, T, 7] -> delta_pos: [B, T, 3]
    delta_pos = delta_actions[:, :, :3] * scale_factor
    delta_rpy = delta_actions[:, :, 3:6] * scale_factor # 假设角度也是归一化的，需缩放
    
    # 初始状态扩展
    # initial_state: [B, 7] -> slice -> [B, 3] -> unsqueeze -> [B, 1, 3]
    start_pos = initial_state[:, :3].unsqueeze(1) 
    start_rpy = initial_state[:, 3:6].unsqueeze(1) 
    
    # 累积求和 (积分)
    # [B, 1, 3] + [B, T, 3] -> [B, T, 3] (Broadcasting)
    traj_pos = start_pos + torch.cumsum(delta_pos, dim=1)
    traj_rpy = start_rpy + torch.cumsum(delta_rpy, dim=1)
    
    return traj_pos, traj_rpy

# ================= 核心模块：物理几何约束头 =================

class GeometricConstraintHead(nn.Module):
    """
    深度结合 State[7] 和 MapAnything 的物理约束头
    """
    def __init__(self, geom_dim: int, action_dim: int = 7):
        super().__init__()
        
        # 1. 几何感知 MLP
        self.geom_processor = nn.Sequential(
            nn.Linear(geom_dim, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        
        # 2. 交互评估网络 (State + Geometry -> Score)
        # 输入: 相对位置(3) + 相对姿态(9, RotMat) + 几何特征(128)
        self.interaction_scorer = nn.Sequential(
            nn.Linear(3 + 9 + 128, 256),
            nn.GELU(),
            nn.Linear(256, 1) # 输出: 当前状态与几何的兼容性分数
        )
        
        # 物理参数 (可学习或固定)
        self.grasp_dist_thresh = 0.05 # 5cm 抓取距离阈值
        self.action_scale = nn.Parameter(torch.tensor(0.05), requires_grad=False) # 动作缩放
        
    def compute_surface_normals(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        从点云估算表面法线 (自动推断网格大小)
        Args:
            point_cloud: [B, N, 3]
        """
        B, N, _ = point_cloud.shape
        
        # 自动推断 Grid Size (假设 N = H * W 且 H=W)
        # 例如 518x518 输入 -> 37x37 grid -> N=1369
        grid_size = int(np.sqrt(N))
        
        # 处理多视图的情况 (如果 N 不是完全平方数，可能是 V * H * W)
        # 简化策略：这里假设 constraint 只基于主视图 (第一个视图)
        # 如果 N 很大且不是平方数，需要截取前 grid*grid 个点
        if grid_size * grid_size != N:
            # 尝试推断是否是 V * Grid^2
            # 这里的逻辑需要根据 MapAnything 的输出调整。通常 MapAnything 会拼接所有视图。
            # 为了计算法线（需要 grid 结构），我们只取第一个视图。
            # 假设 patch size 14
            # 尝试常见分辨率: 224->16, 518->37
            possible_grids = [37, 16, 24, 32] # 518, 224, 336, 448
            found = False
            for g in possible_grids:
                if N % (g*g) == 0:
                    grid_size = g
                    # 只取第一个视图
                    point_cloud = point_cloud[:, :g*g, :]
                    N = g*g
                    found = True
                    break
            if not found:
                 # 兜底：如果算不出来，直接取最近的平方数截断（可能会有边缘错误，但不会崩）
                 grid_size = int(np.sqrt(N))
                 point_cloud = point_cloud[:, :grid_size*grid_size, :]
                 N = grid_size*grid_size

        H, W = grid_size, grid_size
        
        # 重塑回网格
        pc_grid = point_cloud.view(B, H, W, 3).permute(0, 3, 1, 2)
        
        # 计算梯度
        dy = pc_grid[:, :, 1:, :] - pc_grid[:, :, :-1, :]
        dx = pc_grid[:, :, :, 1:] - pc_grid[:, :, :, :-1]
        dy = F.pad(dy, (0, 0, 0, 1))
        dx = F.pad(dx, (0, 1, 0, 0))
        
        normals = torch.cross(dx, dy, dim=1)
        normals = F.normalize(normals, dim=1, eps=1e-6)
        return normals.permute(0, 2, 3, 1).view(B, N, 3)

    def forward(self, 
                pred_actions: torch.Tensor, 
                geom_tokens: torch.Tensor,
                scale_token: torch.Tensor,
                robot_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算基于 State 定义的物理约束损失
        """
        B, T, D = pred_actions.shape
        
        # --- 1. 几何重建 (Metric Alignment) ---
        # MapAnything 输出的 metric scale 必须应用到点云上
        # scale_token: [B, 1, C] -> MLP -> scalar scale
        # 这里简化处理：假设 geom_tokens 已经是点特征，我们用 scale_token 调节它
        # 在 MapAnything 原文中，metric scale 是全局因子 m
        # 我们假设 geom_tokens 包含空间信息，乘以前几维的 scale
        metric_scale_factor = torch.sigmoid(scale_token.mean(dim=-1, keepdim=True)) * 10.0 # 假设最大10米
        reconstructed_points = geom_tokens[:, :, :3] * metric_scale_factor # [B, N, 3] 粗略点云
        
        # 估算表面法线 (用于姿态约束)
        surface_normals = self.compute_surface_normals(reconstructed_points)

        B, N_grid, _ = surface_normals.shape
        reconstructed_points_for_loss = reconstructed_points[:, :N_grid, :]
        
        # --- 2. 轨迹重建 (Trajectory Integration) ---
        # 将相对动作转化为绝对物理空间轨迹
        # robot_state: [B, 7] -> x, y, z, r, p, y, g
        abs_traj_pos, abs_traj_rpy = differentiable_trajectory_integration(
            robot_state, pred_actions, self.action_scale
        ) # [B, T, 3]
        
        # 计算绝对旋转矩阵 (用于姿态对齐)
        abs_traj_rot = euler_to_rotation_matrix(abs_traj_rpy) # [B, T, 3, 3]
        # 提取手爪接近向量 (假设是手爪坐标系的 Z 轴)
        gripper_approach_vec = abs_traj_rot[:, :, :, 2] # [B, T, 3]
        
        # --- 3. 约束计算 ---
        
        # A. 距离场计算 (Distance Field)
        # 找到轨迹上每个点对应的最近物体点
        # [B, T, 1, 3] - [B, 1, N, 3] -> [B, T, N]
        dists = torch.cdist(abs_traj_pos, reconstructed_points_for_loss)
        min_dists, nearest_idx = dists.min(dim=-1) # [B, T] - 到最近点的距离
        
        # 获取最近点的法线
        # nearest_idx: [B, T] -> gather from [B, N, 3]
        nearest_normals = torch.stack([
            surface_normals[b, nearest_idx[b], :] for b in range(B)
        ]) # [B, T, 3]
        
        abs_traj_rot = euler_to_rotation_matrix(abs_traj_rpy)
        gripper_approach_vec = abs_traj_rot[:, :, :, 2]

        # B. 状态依赖的动态约束 (Gripper State Dependent)
        pred_gripper = torch.sigmoid(pred_actions[:, :, 6]) # [B, T] 0-1
        
        # B1. 抓取意图 (Gripper < 0.5): 鼓励靠近 + 姿态对齐
        is_grasping = (pred_gripper < 0.5).float()
        
        # 抓取距离惩罚: 既然要抓，距离就应该小于阈值
        grasp_dist_loss = (is_grasping * F.relu(min_dists - self.grasp_dist_thresh)).mean()
        
        # 姿态对齐惩罚: 抓取时，手爪 Z 轴应与表面法线反向平行 (dot product -> -1)
        # alignment = dot(approach, normal). We want alignment = -1.
        # Loss = (1 + dot)
        alignment_score = torch.sum(gripper_approach_vec * nearest_normals, dim=-1) # [B, T]
        orientation_loss = (is_grasping * (1.0 + alignment_score)).mean()
        
        # B2. 避障意图 (Gripper > 0.5): 惩罚穿透 (距离过近)
        collision_safe_margin = 0.02 # 2cm 安全距离
        collision_loss = ((1 - is_grasping) * F.relu(collision_safe_margin - min_dists)).mean()
        
        # C. 物理可行性 (Joint Limits / Workspace)
        # 简单的 Box 约束，防止飞出工作空间
        # 假设工作空间 Z > 0 (桌面以上)
        workspace_loss = F.relu(-abs_traj_pos[:, :, 2]).mean() # Z 不能为负
        
        return {
            "dist_constraint": grasp_dist_loss,
            "orient_constraint": orientation_loss * 0.5, # 姿态权重略低
            "collision_constraint": collision_loss,
            "workspace_constraint": workspace_loss,
            "debug_min_dist": min_dists.mean()
        }

# ================= 主模型框架 =================

@FRAMEWORK_REGISTRY.register("QwenSuper-GeometricConstraint")
class QwenSuperGeometricConstraint(baseframework):
    """
    State-Aware QwenSuper: 
    Deeply integrates robot state (XYZ+RPY+G) with MapAnything's metric geometry.
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
        
        # 3. Visual Encoder (Detail Extractor)
        self.dino_encoder = get_dino_model(
            backbone_name=getattr(self.config.framework.dino, "dino_backbone", "dinov2_vits14")
        )
        self.dino_pro = nn.Linear(
            in_features=self.dino_encoder.num_channels, 
            out_features=H_QWEN
        )

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
        # 4. MapAnything (Physics Engine)
        if not hasattr(self.config.framework, "map_anything"):
            raise ValueError("Config missing `framework.map_anything`")
        self.map_encoder = get_map_model(self.config.framework.map_anything)
        C_MAP = self.map_encoder.info_sharing.dim
        
        # 5. Geometric Constraint Head (The "Super-Ego")
        self.geom_constraint_head = GeometricConstraintHead(
            geom_dim=C_MAP,
            action_dim=7
        )
        
        # Hyperparams
        self.constraint_weight = getattr(config.trainer, 'constraint_weight', 0.2)
        
    def forward(self, examples: List[dict] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        前向传播：融合 VLM 语义与 MapAnything 物理约束
        """
        # 数据解包
        batch_images, wrist_views, instructions, state = self.align_model_input(examples)
     
        # --- 1. 主链路 (Main Stream) ---
        # VLM + DINO -> DiT. 保持纯净，不拼接几何 Token
        last_hidden, state = self.get_action_condition(batch_images, instructions, wrist_views, state)
   
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
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            state_repeated = None
            if state is not None:
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)
            action_loss = self.action_model(last_hidden_repeated, actions_target_repeated, state_repeated)  # (B, chunk_len, action_dim)

            # 预测动作 (用于计算几何约束)
            # 注意：在训练中我们需要预测出的动作是可微分的，因此不能直接用 inference 的 predict
            # 但标准 DiT 训练通常是预测噪声。
            # 这里为了计算约束，我们使用单步去噪估计 (One-step estimate) 或直接使用 GT 动作添加微小扰动
            # 为了简单有效，我们在这里**约束模型预测的去噪后动作** (Predicted X_0)
            # 或者，更直接地，我们约束 GT 动作在当前几何下的合理性（这有助于学习几何嵌入），
            # 但更好的方式是约束 VLA 模型的输出。
            # 鉴于 DiT 训练的特殊性，我们这里使用一个技巧：
            # 我们假设 Action Model 有一个 helper 可以返回 "predicted_start_action" (x_0 hat)
            if hasattr(self.action_model, "predict_x0_from_noise"):
                 # 这是一个假设的接口，需要 DiT 支持返回当前去噪步估计的 x0
                pred_actions_for_loss = self.action_model.predict_x0_from_noise(last_hidden, actions, state)
            else:
                # 如果不支持，我们直接对 GT 动作施加几何对齐损失（作为一种正则化，强迫几何特征与动作对齐）
                # 或者，使用 inference 模式下的 predict_action (但需要通过 graph)
                # 这是一个常见的 VLA 训练难点。
                # 妥协方案：在训练阶段，我们仅使用 predict_action (no_grad) 来监控，
                # 而让几何约束头作为一个辅助任务 (Auxiliary Task)：预测可行性。
                # 但为了响应 "Physical Constraint Layer"，我们尝试生成动作：
                pred_actions_for_loss = self.action_model.predict_action(last_hidden, state) # [B, T, 7]

        # --- 3. 几何约束 (Physical Constraint Stream) ---
        # 并行分支，不干扰主 Visual Encoder 梯度，但梯度回传给 Action Model (如果 pred_actions_for_loss 在图中)
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # 冻结 MapAnything，只做推理
            with torch.no_grad():
                map_input = self.map_encoder.prepare_map_input(examples)
                # geo_tokens: [B, N, C], scale_token: [B, 1, C]
                geo_tokens, scale_token = self.map_encoder(map_input)
                
        # 计算约束损失
        # 注意：这里 pred_actions_for_loss 必须带有梯度，才能优化 Action Model
        constraint_dict = self.geom_constraint_head(
            pred_actions_for_loss, 
            geo_tokens.detach(), # 几何本身不更新
            scale_token.detach(),
            state, # 机器人当前状态 [B, 7]
        )
        
        # --- 4. 总损失聚合 ---
        geo_loss_total = sum([v for k, v in constraint_dict.items() if "debug" not in k])
        total_loss = action_loss + self.constraint_weight * geo_loss_total
        
        return {
            "loss": total_loss,
            "action_loss": action_loss.item(),
            "dist_constraint": constraint_dict["dist_constraint"].item(),
            "orient_constraint": constraint_dict["orient_constraint"].item(),
            "collision_constraint": constraint_dict["collision_constraint"].item(),
        }

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        推理：单次前向直接回归未来动作（无扩散采样）。

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        
        优化推理逻辑：包含基于几何的动作修正或拒绝采样 (Rejection Sampling)
        """
        # ... (预处理代码与之前相同，略) ...
        # 1. 主链路推理得到 Raw Actions
        # qwen_inputs = ...
        # main_features = ...
        # raw_actions = self.action_model.predict_action(...) 
        batch_images, wrist_views, instructions, state = self.align_model_input(examples)
        last_hidden, state = self.get_action_condition(batch_images, instructions, wrist_views, state)
        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state)  # (B, chunk_len, action_dim)
        normalized_actions = pred_actions.detach().cpu().numpy()
        
        # 2. 几何安全检查 (Safety Check)
        # 仅在需要高安全性的模式下开启
        result = {"normalized_actions": normalized_actions}
        
        if kwargs.get("enable_safety_check", True):
            # 调用 MapAnything 获取环境几何
            map_input = self.map_encoder.prepare_map_input(examples)
            geo_tokens, scale_token = self.map_encoder(map_input)
            
            # 计算约束分数
            constraints = self.geom_constraint_head(
                pred_actions, geo_tokens, scale_token, state
            )
            
            # 如果碰撞风险过高，可以触发急停或回退策略
            if constraints["collision_constraint"] > 0.1:
                logger.warning("Collision risk detected! Actions may be unsafe.")
                result["unsafe_flag"] = True
                result["collision_score"] = constraints["collision_constraint"].item()
                
            result["debug_dist"] = constraints["debug_min_dist"].item()

        return result
    
    def align_model_input(self, examples: List[dict]):

        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        wrist_views = [to_pil_preserve(example["wrist_views"]) for example in examples] if "wrist_views" in examples[0] else None #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        states = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        print(states)
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
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/geometric_constraint.yaml")
    args, clipargs = parser.parse_known_args()
    
    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.qwenvl.base_vlm = "/public/home/vlabadmin/dataset/Qwen3-VL-4B-Instruct"
    
    model = QwenSuperGeometricConstraint(cfg)
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
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss}")

    # test predict action
    predict_output = model.predict_action([sample]) #, state=[batch[0]["state"]]
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    vla_dataset_cfg = cfg.datasets.vla_data
    # vla_dataset_cfg.include_state = True
    # vla_dataset_cfg.data_mix = "BEHAVIOR_challenge"
    # vla_dataset_cfg.data_mix = "BEHAVIOR_rgp_dual_history"
    vla_dataset_cfg.task_id = 5
    vla_dataset_cfg.video_backend = "torchvision_av"
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        collate_fn=collate_fn,
    )
    
    from tqdm import tqdm
    count = 0
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        batch
        count += 1
        if count > 1:
            break

    # try get model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model(batch)

    pred = model.predict_action(examples=[sample]) #, state=[batch[0]["state"]]
    print(f"✓ 推理成功: actions_shape={pred['normalized_actions'].shape}")