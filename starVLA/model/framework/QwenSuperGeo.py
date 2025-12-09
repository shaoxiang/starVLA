# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
Qwen-GeoSuper Framework
A comprehensive geometric-aware VLA model with:
1. Cross-attention geometric fusion
2. Egocentric geometric representation
3. Affordance-aware action prediction
4. Physically-constrained flow matching
"""
from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import einops

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


class GeometricCrossAttentionFusion(nn.Module):
    """几何感知的跨注意力融合模块"""
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 几何到VLM的交叉注意力
        self.geom_to_vlm_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # VLM到几何的交叉注意力（双向交互）
        self.vlm_to_geom_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 尺度感知的门控机制
        self.scale_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid()
        )
        
        # 自适应融合权重
        self.fusion_weights = nn.Parameter(torch.ones(3))  # VLM, Geo-VLM, VLM-Geo
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, vlm_features, geom_features, scale_token=None):
        """
        Args:
            vlm_features: [B, L_vlm, H] VLM特征
            geom_features: [B, L_geo, H] 几何特征
            scale_token: [B, 1, H] 尺度因子token
        Returns:
            fused_features: [B, L_vlm, H] 融合后的特征
            attention_weights: 注意力权重用于可视化
        """
        B = vlm_features.shape[0]
        
        # 1. 几何到VLM的注意力（几何引导语义）
        geom_guided_vlm, attn_weights1 = self.geom_to_vlm_attn(
            query=vlm_features,
            key=geom_features,
            value=geom_features
        )
        geom_guided_vlm = self.norm1(vlm_features + geom_guided_vlm)
        
        # 2. VLM到几何的注意力（语义聚焦几何）
        vlm_guided_geom, attn_weights2 = self.vlm_to_geom_attn(
            query=geom_features,
            key=vlm_features,
            value=vlm_features
        )
        vlm_guided_geom = self.norm2(geom_features + vlm_guided_geom)
        
        # 3. 双向特征融合
        # 3.1 从VLM增强的几何特征中提取全局上下文
        global_geom_context = vlm_guided_geom.mean(dim=1, keepdim=True)  # [B, 1, H]
        
        # 3.2 尺度感知融合
        if scale_token is not None:
            scale_gate = self.scale_gate(scale_token)  # [B, 1, H]
            global_geom_context = global_geom_context * scale_gate
        
        # 3.3 自适应加权融合
        weights = F.softmax(self.fusion_weights, dim=0)
        
        fused = (
            weights[0] * vlm_features +
            weights[1] * geom_guided_vlm +
            weights[2] * global_geom_context.expand_as(vlm_features)
        )
        
        return fused, {'geom_to_vlm': attn_weights1, 'vlm_to_geom': attn_weights2}


class EgocentricGeometricTransform(nn.Module):
    """自我中心几何表示变换"""
    
    def __init__(self, hidden_dim, robot_state_dim=7):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 机器人状态编码器
        self.robot_encoder = nn.Sequential(
            nn.Linear(robot_state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 相对几何变换网络
        self.relative_transform = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        
        # 注意力门控：选择与机器人相关的几何区域
        self.attention_gate = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, global_geom_tokens, robot_state):
        """
        Args:
            global_geom_tokens: [B, L_geo, H] 全局几何tokens
            robot_state: [B, robot_state_dim] 机器人状态
        Returns:
            ego_geom_tokens: [B, L_geo, H] 自我中心几何tokens
        """
        B, L, H = global_geom_tokens.shape
        
        print("global_geom_tokens shape:", global_geom_tokens.shape)
        print("robot_state shape before squeeze:", robot_state.shape)
        
        # 编码机器人状态
        robot_feat = self.robot_encoder(robot_state).unsqueeze(1)  # [B, 1, H]
        
        # 将机器人状态作为查询，提取相关几何
        robot_queries = robot_feat.expand(-1, L, -1)  # [B, L, H]
        
        # 相对几何变换
        combined = torch.cat([robot_queries, global_geom_tokens], dim=0)  # [2B, L, H]
        combined = einops.rearrange(combined, '(b n) l h -> b (n l) h', b=B, n=2)
        
        transformed = self.relative_transform(combined)
        transformed = einops.rearrange(transformed, 'b (n l) h -> (b n) l h', b=B, n=2)
        
        ego_geom = transformed[B:]  # 提取转换后的几何部分
        
        # 注意力门控：强调机器人附近的几何
        attended_geom, _ = self.attention_gate(
            query=robot_feat,
            key=ego_geom,
            value=ego_geom
        )
        
        # 残差连接
        ego_geom = ego_geom + attended_geom.expand_as(ego_geom)
        
        return ego_geom


class AffordanceAwareModule(nn.Module):
    """可供性感知模块"""
    
    def __init__(self, hidden_dim, action_dim, num_affordances=8):
        super().__init__()
        
        self.num_affordances = num_affordances
        
        # 几何可供性预测器
        self.geom_affordance_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_affordances)
        )
        
        # 动作可供性预测器
        self.action_affordance_predictor = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_affordances)
        )
        
        # 可供性一致性损失
        self.affordance_loss_fn = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, geom_features, actions):
        """
        Args:
            geom_features: [B, L_geo, H] 几何特征
            actions: [B, T, action_dim] 预测的动作
        Returns:
            affordance_loss: 可供性一致性损失
            geom_affordance: 几何可供性分布
            action_affordance: 动作可供性分布
        """
        # 提取几何可供性
        geom_affordance = self.geom_affordance_predictor(
            geom_features.mean(dim=1)  # [B, num_affordances]
        )
        geom_affordance = F.log_softmax(geom_affordance, dim=-1)
        
        # 提取动作可供性
        action_affordance = self.action_affordance_predictor(
            actions.mean(dim=1)  # [B, num_affordances]
        )
        action_affordance = F.softmax(action_affordance, dim=-1)
        
        # 计算可供性一致性损失
        affordance_loss = self.affordance_loss_fn(
            geom_affordance,
            action_affordance.detach()  # 只优化几何预测器
        )
        
        # 计算可供性匹配度（用于分析）
        match_score = F.cosine_similarity(
            geom_affordance.exp(),  # 转换为概率
            action_affordance,
            dim=-1
        ).mean()
        
        return {
            'loss': affordance_loss,
            'geom_affordance': geom_affordance.exp(),
            'action_affordance': action_affordance,
            'match_score': match_score
        }


class PhysicallyConstrainedFlowMatching(FlowmatchingActionHead):
    """物理约束的流匹配动作头"""
    
    def __init__(self, full_config):
        super().__init__(full_config)
        
        # 物理约束参数
        action_config = full_config.framework.action_model
        self.collision_threshold = getattr(full_config, 'collision_threshold', 0.05)
        self.velocity_limit = getattr(full_config, 'velocity_limit', 1.0)
        
        # 碰撞预测网络
        self.collision_predictor = nn.Sequential(
            nn.Linear(action_config.hidden_size + action_config.action_dim, 256),  # 特征+动作维度
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 平滑度正则化
        self.smoothness_weight = getattr(full_config, 'smoothness_weight', 0.1)
        
    def compute_physical_constraints(self, actions, geom_features):
        """
        计算物理约束损失
        """
        B, T, D = actions.shape
        
        # 1. 碰撞约束
        # 将动作映射到末端执行器位置（简化版本）
        end_effector_pos = actions[:, :, :3]  # 假设前3维是位置
        
        # 从几何特征中提取物体表面点（简化）
        # 实际中应该从MapAnything的深度图重建点云
        surface_points = geom_features[:, :, :3]  # 假设前3维是3D坐标
        
        # 计算最小距离
        distances = torch.cdist(end_effector_pos, surface_points)  # [B, T, N_points]
        min_distances, _ = distances.min(dim=-1)
        
        # 碰撞惩罚：距离小于阈值时惩罚
        collision_penalty = F.relu(self.collision_threshold - min_distances)
        collision_loss = collision_penalty.mean()
        
        # 2. 平滑度约束
        velocity = actions[:, 1:, :] - actions[:, :-1, :]
        smoothness_loss = velocity.norm(dim=-1).mean()
        
        # 3. 关节极限约束（如果动作包含关节角度）
        if D > 3:
            joint_angles = actions[:, :, 3:6]  # 假设接下来的3维是关节角度
            joint_limit_loss = F.relu(torch.abs(joint_angles) - 1.0).mean()  # 假设限制在[-1,1]
        else:
            joint_limit_loss = torch.tensor(0.0, device=actions.device)
        
        return {
            'collision': collision_loss,
            'smoothness': smoothness_loss * self.smoothness_weight,
            'joint_limit': joint_limit_loss
        }
    
    def forward(self, context, target_actions, state=None):
        # 原始流匹配损失
        base_loss = super().forward(context, target_actions, state)
        
        # 添加物理约束
        if self.training:
            with torch.no_grad():
                # 预测动作序列
                pred_actions = self.predict_action(context, state)
            
            # 计算物理约束损失
            physical_losses = self.compute_physical_constraints(
                pred_actions, 
                context  # 使用上下文特征作为几何信息
            )
            
            total_loss = base_loss
            for name, loss in physical_losses.items():
                total_loss = total_loss + loss * 0.1  # 小权重
            
            return total_loss
        
        return base_loss


@FRAMEWORK_REGISTRY.register("Qwen-GeoSuper")
class QwenGeoSuper(baseframework):
    """
    几何感知的视觉-语言-动作模型
    
    核心组件:
    1. Qwen-VL: 视觉语言理解
    2. DINOv2: 稠密视觉特征
    3. MapAnything: 度量3D几何重建
    4. 几何跨注意力融合: 深度融合几何与语义
    5. 自我中心几何变换: 以机器人为中心的表示
    6. 可供性感知模块: 物理交互一致性
    7. 物理约束流匹配: 安全可行的动作生成
    """
    
    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        
        # ================== 基础编码器 ==================
        # 1. VLM编码器
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        
        # 2. DINO编码器
        self.dino_encoder = get_dino_model(
            backbone_name=getattr(self.config.framework.dino, "dino_backbone", "dinov2_vits14")
        )
        self.dino_pro = nn.Linear(
            in_features=self.dino_encoder.num_channels,
            out_features=self.qwen_vl_interface.model.config.hidden_size
        )
        
        # 3. MapAnything编码器
        if not hasattr(self.config.framework, "map_anything"):
            raise ValueError("Qwen-GeoSuper: config中缺少 `framework.map_anything` 配置。")
        self.map_encoder = get_map_model(self.config.framework.map_anything)
        
        # 动态获取维度
        C_MAP = self.map_encoder.info_sharing.dim  # MapAnything特征维度
        H_QWEN = self.qwen_vl_interface.model.config.hidden_size  # VLM隐藏维度
        
        # MapAnything投影层
        self.map_patch_pro = nn.Linear(C_MAP, H_QWEN)
        self.map_scale_pro = nn.Linear(C_MAP, H_QWEN)
        
        # ================== 几何感知模块 ==================
        # 4. 自我中心几何变换
        self.ego_geom_transform = EgocentricGeometricTransform(
            hidden_dim=H_QWEN,
            robot_state_dim=getattr(config.framework, 'robot_state_dim', 7)
        )
        
        # 5. 几何跨注意力融合
        self.geometric_fusion = GeometricCrossAttentionFusion(
            hidden_dim=H_QWEN,
            num_heads=getattr(config.framework, 'num_attention_heads', 8)
        )
        
        # 6. 可供性感知模块
        self.affordance_module = AffordanceAwareModule(
            hidden_dim=H_QWEN,
            action_dim=getattr(config.framework.action_model, 'action_dim', 7),
            num_affordances=getattr(config.framework, 'num_affordances', 8)
        )
        
        # ================== 动作生成模块 ==================
        # 7. 物理约束流匹配头
        # 调整配置以支持跨注意力维度
        if hasattr(config.framework.action_model, 'diffusion_model_cfg'):
            config.framework.action_model.diffusion_model_cfg.cross_attention_dim = H_QWEN

        # 使用改进的流匹配头
        self.action_model = PhysicallyConstrainedFlowMatching(
            config
        )
        
        # ================== 其他配置 ==================
        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
        # 损失权重
        self.loss_weights = {
            'action': getattr(config.trainer, 'action_loss_weight', 1.0),
            'affordance': getattr(config.trainer, 'affordance_loss_weight', 0.1),
            'physical': getattr(config.trainer, 'physical_loss_weight', 0.05)
        }
        
        # 训练阶段管理（用于课程学习）
        self.training_stage = 0
        self.max_training_stage = getattr(config.trainer, 'max_training_stages', 3)
        
    def set_training_stage(self, stage):
        """设置当前训练阶段（用于课程学习）"""
        self.training_stage = stage
        logger.info(f"切换到训练阶段 {stage}/{self.max_training_stage}")
        
        # 根据训练阶段调整学习率和损失权重
        if stage == 0:  # 第一阶段：基础训练
            self.loss_weights['affordance'] = 0.05
            self.loss_weights['physical'] = 0.01
        elif stage == 1:  # 第二阶段：加入几何约束
            self.loss_weights['affordance'] = 0.1
            self.loss_weights['physical'] = 0.05
        else:  # 第三阶段：全约束训练
            self.loss_weights['affordance'] = 0.15
            self.loss_weights['physical'] = 0.1
    
    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        训练前向传播
        
        流程:
        1. 编码多模态输入
        2. 几何感知融合
        3. 可供性一致性监督
        4. 物理约束动作生成
        """
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        states = [example["state"] for example in examples] if "state" in examples[0] else None
        
        # ================== 步骤1: 多模态编码 ==================
        # 1.1 VLM编码
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, 
            instructions=instructions
        )
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            vlm_features = qwenvl_outputs.hidden_states[-1]  # [B, L_vlm, H]
            
            # 1.2 DINO编码
            image_tensors = self.dino_encoder.prepare_dino_input(batch_images)
            B = len(batch_images)
            dino_features = self.dino_encoder(image_tensors)  # [B*num_view, tokens, dim]
            dino_features = dino_features.reshape(B, -1, dino_features.shape[-1])
            dino_features = self.dino_pro(dino_features)  # [B, num_view*tokens, H]
            
            # 1.3 MapAnything编码
            map_input_list = self.map_encoder.prepare_map_input(examples)
            map_patch_tokens, map_scale_token = self.map_encoder(map_input_list)
            
            # 投影到统一空间
            map_patch_tokens_pro = self.map_patch_pro(map_patch_tokens)
            map_scale_token_pro = self.map_scale_pro(map_scale_token)
            
            # ================== 步骤2: 几何感知融合 ==================
            # 2.1 自我中心几何变换（如果需要机器人状态）
            if states is not None:
                print("states shape:", np.array(states).shape, states)

                robot_states = torch.tensor(
                    np.array([s[0, :] for s in states]),  # 取当前状态
                    device=map_patch_tokens_pro.device,
                    dtype=map_patch_tokens_pro.dtype
                ) # 形状为 [B, 7]（此处B=2）

                ego_geom_tokens = self.ego_geom_transform(
                    map_patch_tokens_pro, 
                    robot_states
                )
            else:
                ego_geom_tokens = map_patch_tokens_pro
            
            # 2.2 几何跨注意力融合
            fused_vlm_features, attention_info = self.geometric_fusion(
                vlm_features,
                ego_geom_tokens,
                map_scale_token_pro
            )
            
            # 2.3 最终特征拼接
            # 保留VLM特征用于语言理解，加入DINO的稠密特征
            combined_features = torch.cat([
                fused_vlm_features,           # 几何增强的VLM特征
                dino_features,                # 稠密视觉特征
                ego_geom_tokens,              # 自我中心几何特征
                map_scale_token_pro           # 全局尺度因子
            ], dim=1)
            
            # ================== 步骤3: 动作预测 ==================
            # 3.1 准备动作标签
            actions_tensor = torch.tensor(
                np.array(actions),
                device=combined_features.device,
                dtype=combined_features.dtype
            )
            actions_target = actions_tensor[:, -(self.future_action_window_size+1):, :]
            
            # 3.2 重复采样用于流匹配训练
            repeated_diffusion_steps = self.config.trainer.get("repeated_diffusion_steps", 4)
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            features_repeated = combined_features.repeat(repeated_diffusion_steps, 1, 1)
            
            states_repeated = None
            if states is not None:
                states_tensor = torch.tensor(
                    np.array(states),
                    device=combined_features.device,
                    dtype=combined_features.dtype
                )
                states_repeated = states_tensor.repeat(repeated_diffusion_steps, 1, 1)
            
            # 3.3 流匹配损失
            action_loss = self.action_model(
                features_repeated,
                actions_target_repeated,
                states_repeated
            )
            
            # ================== 步骤4: 辅助损失 ==================
            # 4.1 可供性一致性损失
            # 使用预测的动作计算可供性损失
            with torch.no_grad():
                pred_actions = self.action_model.predict_action(
                    combined_features, 
                    states_tensor if states is not None else None
                )
            
            affordance_info = self.affordance_module(
                ego_geom_tokens,
                pred_actions
            )
            affordance_loss = affordance_info['loss']
            
            # ================== 步骤5: 总损失 ==================
            total_loss = (
                self.loss_weights['action'] * action_loss +
                self.loss_weights['affordance'] * affordance_loss
            )
        
        return {
            'loss': total_loss,
            'action_loss': action_loss,
            'affordance_loss': affordance_loss,
            'affordance_match_score': affordance_info['match_score'],
            'attention_info': attention_info
        }
    
    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        robot_state: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        推理：生成未来动作序列
        
        支持额外的机器人状态输入，用于自我中心几何变换
        """
        # 图像尺寸调整
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        
        B = len(batch_images)
        
        # ================== 步骤1: 多模态编码 ==================
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions
        )
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            vlm_features = qwenvl_outputs.hidden_states[-1]
            
            # DINO编码
            image_tensors = self.dino_encoder.prepare_dino_input(batch_images)
            dino_features = self.dino_encoder(image_tensors)
            dino_features = dino_features.reshape(B, -1, dino_features.shape[-1])
            dino_features = self.dino_pro(dino_features)
            
            # MapAnything编码
            # 构建推理示例
            predict_examples = [
                {
                    "image": batch_images[i],
                    "lang": instructions[i]
                } for i in range(B)
            ]
            
            # 添加可选的机器人输入
            if robot_state is not None:
                for i in range(B):
                    predict_examples[i]["robot_state"] = robot_state[i]
            
            map_input_list = self.map_encoder.prepare_map_input(predict_examples)
            map_patch_tokens, map_scale_token = self.map_encoder(map_input_list)
            
            map_patch_tokens_pro = self.map_patch_pro(map_patch_tokens)
            map_scale_token_pro = self.map_scale_pro(map_scale_token)
            
            # ================== 步骤2: 几何感知融合 ==================
            # 自我中心几何变换
            if robot_state is not None:
                robot_state_tensor = torch.from_numpy(np.array(robot_state)).to(
                    map_patch_tokens_pro.device,
                    dtype=map_patch_tokens_pro.dtype
                )
                ego_geom_tokens = self.ego_geom_transform(
                    map_patch_tokens_pro,
                    robot_state_tensor
                )
            else:
                ego_geom_tokens = map_patch_tokens_pro
            
            # 几何跨注意力融合
            fused_vlm_features, attention_info = self.geometric_fusion(
                vlm_features,
                ego_geom_tokens,
                map_scale_token_pro
            )
            
            # 最终特征拼接
            combined_features = torch.cat([
                fused_vlm_features,
                dino_features,
                ego_geom_tokens,
                map_scale_token_pro
            ], dim=1)
        
        # ================== 步骤3: 动作生成 ==================
        state_tensor = None
        if state is not None:
            state_tensor = torch.from_numpy(np.array(state)).to(
                combined_features.device,
                dtype=combined_features.dtype
            )
        
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                combined_features,
                state_tensor
            )
        
        # ================== 步骤4: 后处理和分析 ==================
        # 可供性分析
        affordance_info = self.affordance_module(
            ego_geom_tokens,
            pred_actions
        )
        
        # 转换为numpy
        normalized_actions = pred_actions.detach().cpu().numpy()
        
        return {
            'normalized_actions': normalized_actions,
            'affordance_match_score': affordance_info['match_score'].item(),
            'attention_weights': {
                k: v.detach().cpu().numpy() for k, v in attention_info.items()
            },
            'ego_geom_tokens': ego_geom_tokens.detach().cpu().numpy(),
            'scale_factor': map_scale_token_pro.detach().cpu().to(torch.float32).numpy()
        }
    
    def visualize_attention(self, images, attention_weights):
        """
        可视化注意力权重
        
        Args:
            images: 原始图像
            attention_weights: 注意力权重字典
        Returns:
            可视化图像列表
        """
        import matplotlib.pyplot as plt
        import cv2
        
        visualizations = []
        
        for i, img in enumerate(images):
            if isinstance(img, list):
                img = img[0]  # 取第一个视角
            
            img_np = np.array(img)
            if len(img_np.shape) == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # 绘制几何到VLM的注意力热图
            geom_to_vlm_attn = attention_weights['geom_to_vlm'][i]
            
            # 平均注意力权重
            avg_attn = geom_to_vlm_attn.mean(axis=0)
            
            # 调整大小以匹配图像
            h, w = img_np.shape[:2]
            attn_map = cv2.resize(avg_attn, (w, h))
            
            # 创建热图
            heatmap = cv2.applyColorMap(
                np.uint8(255 * attn_map / attn_map.max()),
                cv2.COLORMAP_JET
            )
            
            # 融合热图和原图
            overlayed = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
            visualizations.append(overlayed)
        
        return visualizations


class DevelopmentalCurriculumTrainer:
    """发展式课程学习训练器"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.current_stage = 0
        
        # 定义训练阶段
        self.stages = [
            {
                'name': 'stage0_baseline',
                'freeze_modules': ['map_encoder', 'affordance_module'],
                'loss_weights': {'action': 1.0, 'affordance': 0.0, 'physical': 0.0},
                'max_epochs': 5
            },
            {
                'name': 'stage1_geometric_fusion',
                'freeze_modules': [],
                'loss_weights': {'action': 1.0, 'affordance': 0.1, 'physical': 0.05},
                'max_epochs': 10
            },
            {
                'name': 'stage2_full_constraints',
                'freeze_modules': [],
                'loss_weights': {'action': 1.0, 'affordance': 0.15, 'physical': 0.1},
                'max_epochs': 15
            }
        ]
    
    def set_stage(self, stage_idx):
        """设置当前训练阶段"""
        if stage_idx >= len(self.stages):
            return False
        
        self.current_stage = stage_idx
        stage_config = self.stages[stage_idx]
        
        # 冻结指定模块
        for name, module in self.model.named_modules():
            if any(frozen in name for frozen in stage_config['freeze_modules']):
                for param in module.parameters():
                    param.requires_grad = False
                logger.info(f"冻结模块: {name}")
        
        # 设置损失权重
        self.model.loss_weights = stage_config['loss_weights']
        self.model.set_training_stage(stage_idx)
        
        logger.info(f"切换到训练阶段: {stage_config['name']}")
        return True
    
    def should_advance_stage(self, success_rate, current_epoch):
        """判断是否应该进入下一阶段"""
        stage_config = self.stages[self.current_stage]
        
        # 检查是否达到最大epoch
        if current_epoch >= stage_config['max_epochs']:
            return True
        
        # 检查成功率是否达标
        if success_rate > 0.8:  # 80%成功率
            return True
        
        return False


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./starVLA/config/training/geosuper.yaml",
        help="Path to YAML config"
    )
    args, clipargs = parser.parse_known_args()
    
    # 加载配置
    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.qwenvl.base_vlm = "/public/home/vlabadmin/dataset/Qwen3-VL-4B-Instruct"
    
    # 创建模型
    model = QwenGeoSuper(cfg)
    print(f"模型总参数量: {sum(p.numel() for p in model.parameters())}")
    print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 创建课程学习训练器
    curriculum_trainer = DevelopmentalCurriculumTrainer(model, cfg)
    
    # 测试数据
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float32),
        "image": [image, image],
        "lang": "Pick up the red cup and place it on the table.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float32),
    }
    
    batch = [sample, sample]
    
    # 测试训练前向
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    
    # 第一阶段训练
    curriculum_trainer.set_stage(0)
    forward_output = model(batch)
    print(f"第一阶段总损失: {forward_output['loss'].item():.4f}")
    print(f"动作损失: {forward_output['action_loss'].item():.4f}")
    print(f"可供性匹配度: {forward_output['affordance_match_score'].item():.4f}")
    
    # 第二阶段训练
    curriculum_trainer.set_stage(1)
    forward_output = model(batch)
    print(f"\n第二阶段总损失: {forward_output['loss'].item():.4f}")
    
    # 测试推理
    model.eval()
    predict_output = model.predict_action(
        batch_images=[batch[0]["image"]],
        instructions=[batch[0]["lang"]],
        state=[batch[0]["state"]],
        robot_state=np.random.uniform(-1, 1, size=(1, 7)).astype(np.float32)
    )
    
    print(f"\n预测动作形状: {predict_output['normalized_actions'].shape}")
    print(f"可供性匹配分数: {predict_output['affordance_match_score']:.4f}")
    print(f"尺度因子: {predict_output['scale_factor'].squeeze()}")
    
    # 可视化注意力
    if 'attention_weights' in predict_output:
        print(f"注意力权重形状: {predict_output['attention_weights']['geom_to_vlm'].shape}")