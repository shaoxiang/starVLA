# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
Qwen-GeoSuper 训练脚本
支持课程学习和几何感知模块
"""

import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
import yaml
import numpy as np
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import get_scheduler

# Local imports
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups
from starVLA.dataloader import build_dataloader

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 初始化加速器
deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
logger = get_logger(__name__)


class CurriculumLearningManager:
    """课程学习管理器"""
    
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        self.current_stage = getattr(cfg.trainer, 'curriculum_stage', 0)
        self.training_strategy = getattr(cfg.trainer, 'training_strategy', 'curriculum')
        
        # 获取课程学习配置
        if hasattr(cfg.trainer, 'curriculum_learning') and cfg.trainer.curriculum_learning.enabled:
            self.curriculum_enabled = True
            self.stages = cfg.trainer.curriculum_learning.stages
            self.num_stages = len(self.stages)
        else:
            self.curriculum_enabled = False
            self.stages = []
            self.num_stages = 0
        
        # 打印课程学习配置
        if accelerator.is_main_process:
            logger.info(f"课程学习管理器初始化: 策略={self.training_strategy}, 阶段={self.current_stage}/{self.num_stages}")
    
    def set_stage(self, stage: int):
        """设置当前训练阶段"""
        if not self.curriculum_enabled or stage >= self.num_stages:
            return
        
        self.current_stage = stage
        stage_config = self.stages[stage]
        
        if accelerator.is_main_process:
            logger.info(f"切换到训练阶段 {stage}: {stage_config.name}")
        
        # 冻结/解冻模块
        self._apply_freezing(stage_config.freeze_modules)
        
        # 设置损失权重
        if hasattr(self.model, 'set_training_stage'):
            self.model.set_training_stage(stage)
    
    def _apply_freezing(self, freeze_modules: List[str]):
        """应用模块冻结策略"""
        if not freeze_modules:
            if accelerator.is_main_process:
                logger.info("无冻结模块")
            return
        
        # 将冻结模块列表转换为集合
        freeze_set = set([m.strip() for m in freeze_modules if m.strip()])
        
        # 遍历模型参数，设置 requires_grad
        for name, param in self.model.named_parameters():
            should_freeze = False
            for module_name in freeze_set:
                if module_name in name:
                    should_freeze = True
                    break
            
            param.requires_grad = not should_freeze
        
        # 打印冻结信息
        if accelerator.is_main_process:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"可训练参数: {trainable_params}/{total_params} ({trainable_params/total_params*100:.1f}%)")
    
    def should_advance_stage(self, step: int, success_rate: float = 0.0) -> bool:
        """判断是否应该进入下一阶段"""
        if not self.curriculum_enabled:
            return False
        
        if self.current_stage >= self.num_stages - 1:
            return False
        
        current_stage_config = self.stages[self.current_stage]
        
        # 检查步数条件
        if step >= current_stage_config.max_steps:
            if accelerator.is_main_process:
                logger.info(f"达到阶段 {self.current_stage} 的最大步数 {step}/{current_stage_config.max_steps}")
            return True
        
        # 检查成功率条件（如果提供了）
        if success_rate > 0.8:  # 80%成功率
            if accelerator.is_main_process:
                logger.info(f"阶段 {self.current_stage} 达到成功率 {success_rate:.2f}")
            return True
        
        return False
    
    def get_loss_weights(self) -> Dict[str, float]:
        """获取当前阶段的损失权重"""
        if not self.curriculum_enabled or self.current_stage >= self.num_stages:
            return {"action": 1.0, "affordance": 0.0, "physical": 0.0}
        
        return self.stages[self.current_stage].loss_weights


class GeoSuperTrainer(TrainerUtils):
    """Qwen-GeoSuper 训练器"""
    
    def __init__(self, cfg, model, train_dataloader, optimizer, lr_scheduler, accelerator):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        
        # 训练状态
        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        
        # 课程学习管理器
        self.curriculum_manager = CurriculumLearningManager(cfg, model)
        
        # 几何感知监控
        self.enable_geometric_debug = getattr(cfg, 'enable_geometric_debug', False)
        self.enable_attention_visualization = getattr(cfg, 'enable_attention_visualization', False)
        
        # 检查点目录
        self.checkpoint_dir = None
        
    def _calculate_total_batch_size(self):
        """计算全局批次大小"""
        return (
            self.cfg.datasets.vla_data.per_device_batch_size *
            self.accelerator.num_processes *
            self.accelerator.gradient_accumulation_steps
        )
    
    def prepare_training(self):
        """准备训练"""
        # 设置随机种子
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.cfg.seed + rank if hasattr(self.cfg, 'seed') else rank + 3047
        set_seed(seed)
        
        # 加载预训练检查点
        if hasattr(self.cfg.trainer, 'pretrained_checkpoint') and self.cfg.trainer.pretrained_checkpoint:
            self._load_pretrained_checkpoint()
        
        # 应用课程学习阶段
        self.curriculum_manager.set_stage(self.cfg.trainer.curriculum_stage)
        
        # 打印可训练参数
        self.print_trainable_parameters(self.model)
        
        # 分布式训练设置
        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )
        
        # 初始化WandB
        if self.accelerator.is_main_process:
            self._init_wandb()
        
        # 初始化检查点目录
        self.checkpoint_dir = Path(self.cfg.output_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if accelerator.is_main_process:
            logger.info("训练准备完成")
    
    def _init_wandb(self):
        """初始化WandB"""
        wandb.init(
            name=self.cfg.run_id,
            dir=os.path.join(self.cfg.output_dir, "wandb"),
            project=self.cfg.wandb_project,
            entity=self.cfg.wandb_entity,
            group="geosuper-train",
            config=OmegaConf.to_container(self.cfg, resolve=True)
        )
    
    def _load_pretrained_checkpoint(self):
        """加载预训练检查点"""
        checkpoint_path = self.cfg.trainer.pretrained_checkpoint
        if accelerator.is_main_process:
            logger.info(f"加载预训练检查点: {checkpoint_path}")
        
        # 加载模型状态
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 选择性加载模块
        reload_modules = getattr(self.cfg.trainer, 'reload_modules', None)
        if reload_modules:
            reload_set = set([m.strip() for m in reload_modules.split(',') if m.strip()])
            model_state_dict = self.model.state_dict()
            
            # 只加载指定的模块
            for name, param in checkpoint.items():
                for module_name in reload_set:
                    if module_name in name:
                        model_state_dict[name] = param
                        break
            
            self.model.load_state_dict(model_state_dict, strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)
    
    def train(self):
        """训练循环"""
        self._log_training_config()
        
        # 创建数据迭代器
        data_iter = iter(self.train_dataloader)
        
        # 创建进度条
        progress_bar = tqdm(
            range(self.cfg.trainer.max_train_steps),
            disable=not self.accelerator.is_local_main_process
        )

        logger.info(f"max_train_steps: {self.cfg.trainer.max_train_steps}, curriculum_stage: {self.cfg.trainer.curriculum_stage}")
        logger.info(f"train_dataloader length: {len(self.train_dataloader)}")

        # 主训练循环
        while self.completed_steps < self.cfg.trainer.max_train_steps:
            # 获取数据批次
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
            
            # 训练步骤
            step_metrics = self._train_step(batch)
            
            # 更新进度
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1
            
            # 更新进度条描述
            if self.accelerator.is_local_main_process:
                loss_info = {
                    "loss": f"{step_metrics.get('total_loss', 0):.4f}",
                    "action": f"{step_metrics.get('action_loss', 0):.4f}",
                    "affordance": f"{step_metrics.get('affordance_loss', 0):.4f}",
                    "lr": f"{self.lr_scheduler.get_last_lr()[0]:.2e}",
                }
                progress_bar.set_postfix(loss_info)
            
            # 记录指标
            if self.completed_steps % self.cfg.trainer.logging_frequency == 0:
                self._log_metrics(step_metrics)
            
            # 评估
            if self.completed_steps % self.cfg.trainer.eval_interval == 0:
                self._evaluate_model()
            
            # 保存检查点
            if self.completed_steps % self.cfg.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()
            
            # 检查课程学习阶段切换
            if self.curriculum_manager.should_advance_stage(self.completed_steps):
                self._advance_curriculum_stage()
            
            # 检查终止条件
            if self.completed_steps >= self.cfg.trainer.max_train_steps:
                break
        
        # 训练结束
        self._finalize_training()
    
    def _train_step(self, batch):
        """单个训练步骤"""
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            
            # 前向传播
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output_dict = self.model(batch)
                
                # 获取损失权重
                loss_weights = self.curriculum_manager.get_loss_weights()
                
                # 计算加权总损失
                total_loss = (
                    output_dict.get('action_loss', 0) * loss_weights['action'] +
                    output_dict.get('affordance_loss', 0) * loss_weights['affordance'] +
                    output_dict.get('physical_constraint_loss', 0) * loss_weights.get('physical', 0)
                )
            
            # 反向传播
            self.accelerator.backward(total_loss)
            
            # 梯度裁剪
            if self.cfg.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.trainer.gradient_clipping
                )
            
            # 优化器步骤
            self.optimizer.step()
            self.lr_scheduler.step()
        
        # 收集指标
        metrics = {
            'total_loss': total_loss.item(),
            'action_loss': output_dict.get('action_loss', torch.tensor(0)).item(),
            'affordance_loss': output_dict.get('affordance_loss', torch.tensor(0)).item(),
            'affordance_match_score': output_dict.get('affordance_match_score', torch.tensor(0)).item(),
        }
        
        # 添加物理约束损失（如果有）
        if 'physical_constraint_loss' in output_dict:
            metrics['physical_constraint_loss'] = output_dict['physical_constraint_loss'].item()
        
        return metrics
    
    def _log_metrics(self, metrics):
        """记录指标到WandB"""
        if self.accelerator.is_main_process:
            # 添加学习率
            metrics['learning_rate'] = self.lr_scheduler.get_last_lr()[0]
            
            # 添加训练阶段
            metrics['curriculum_stage'] = self.curriculum_manager.current_stage
            
            # 记录到WandB
            wandb.log(metrics, step=self.completed_steps)
            
            # 打印日志
            logger.info(f"步骤 {self.completed_steps}: {metrics}")
    
    def _evaluate_model(self):
        """评估模型"""
        if not self.accelerator.is_main_process:
            return
        
        try:
            # 获取评估批次
            data_iter = iter(self.train_dataloader)
            batch = next(data_iter)
            
            # 切换到评估模式
            self.model.eval()
            
            with torch.no_grad():
                # 预测动作
                batch_images = [example["image"] for example in batch]
                instructions = [example["lang"] for example in batch]
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    output_dict = self.model.predict_action(
                        batch_images=batch_images,
                        instructions=instructions,
                        use_ddim=True,
                        num_ddim_steps=20
                    )
                
                # 计算评估指标
                if 'normalized_actions' in output_dict:
                    pred_actions = output_dict['normalized_actions']
                    gt_actions = np.array([example["action"] for example in batch])
                    
                    # 计算MSE
                    mse = ((pred_actions - gt_actions) ** 2).mean()
                    
                    # 记录到WandB
                    wandb.log({
                        'eval_mse': mse,
                        'eval_affordance_match': output_dict.get('affordance_match_score', 0),
                    }, step=self.completed_steps)
            
            # 切换回训练模式
            self.model.train()
            
        except Exception as e:
            logger.warning(f"评估失败: {e}")
    
    def _save_checkpoint(self):
        """保存检查点"""
        if self.accelerator.is_main_process:
            checkpoint_path = self.checkpoint_dir / f"steps_{self.completed_steps}"
            
            # 保存模型状态
            model_state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(model_state_dict, f"{checkpoint_path}_model.pt")
            
            # 保存优化器状态
            optimizer_state_dict = self.optimizer.state_dict()
            torch.save(optimizer_state_dict, f"{checkpoint_path}_optimizer.pt")
            
            # 保存训练元数据
            metadata = {
                'completed_steps': self.completed_steps,
                'curriculum_stage': self.curriculum_manager.current_stage,
                'timestamp': time.time(),
            }
            torch.save(metadata, f"{checkpoint_path}_metadata.pt")
            
            logger.info(f"检查点已保存: {checkpoint_path}")
    
    def _advance_curriculum_stage(self):
        """进入下一训练阶段"""
        new_stage = self.curriculum_manager.current_stage + 1
        self.curriculum_manager.set_stage(new_stage)
        
        # 记录阶段切换
        if self.accelerator.is_main_process:
            logger.info(f"进入课程学习阶段 {new_stage}")
            wandb.log({'curriculum_stage': new_stage}, step=self.completed_steps)
    
    def _log_training_config(self):
        """记录训练配置"""
        if self.accelerator.is_main_process:
            logger.info("***** 训练配置 *****")
            logger.info(f"  总优化步骤: {self.cfg.trainer.max_train_steps}")
            logger.info(f"  每设备批次大小: {self.cfg.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  梯度累积步骤: {self.cfg.trainer.gradient_accumulation_steps}")
            logger.info(f"  总批次大小: {self.total_batch_size}")
            logger.info(f"  训练策略: {self.cfg.trainer.training_strategy}")
            logger.info(f"  课程学习阶段: {self.cfg.trainer.curriculum_stage}")
    
    def _finalize_training(self):
        """训练结束处理"""
        # 保存最终模型
        if self.accelerator.is_main_process:
            final_dir = Path(self.cfg.output_dir) / "final_model"
            final_dir.mkdir(exist_ok=True)
            
            model_state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(model_state_dict, final_dir / "model.pt")
            
            logger.info(f"训练完成，最终模型保存在: {final_dir}")
        
        # 关闭WandB
        if self.accelerator.is_main_process:
            wandb.finish()
        
        self.accelerator.wait_for_everyone()


def setup_directories(cfg) -> Path:
    """设置输出目录"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)
    
    if not dist.is_initialized() or dist.get_rank() == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)
        
        # 保存配置
        OmegaConf.save(cfg, output_dir / "config.yaml")
        with open(output_dir / "config.yaml", "r") as f_yaml, open(output_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)
    
    return output_dir


def build_model(cfg):
    """构建模型"""
    logger.info(f"构建模型: {cfg.framework.name}")
    model = build_framework(cfg)
    return model


def prepare_data(cfg):
    """准备数据"""
    logger.info(f"创建VLA数据加载器: {cfg.datasets.vla_data.data_mix}")
    train_dataloader = build_dataloader(
        cfg=cfg,
        dataset_py=cfg.datasets.vla_data.dataset_py
    )
    return train_dataloader


def setup_optimizer_and_scheduler(model, cfg):
    """设置优化器和学习率调度器"""
    # 参数分组
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )
    
    # 学习率调度器
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )
    
    return optimizer, lr_scheduler


def main(cfg):
    """主函数"""
    logger.info("Qwen-GeoSuper 训练启动")
    
    # 设置目录
    output_dir = setup_directories(cfg)
    
    # 构建模型
    model = build_model(cfg)
    
    # 准备数据
    train_dataloader = prepare_data(cfg)
    
    # 设置优化器和调度器
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model, cfg)
    
    # 创建训练器
    trainer = GeoSuperTrainer(
        cfg=cfg,
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )
    
    # 准备训练
    trainer.prepare_training()
    
    # 开始训练
    trainer.train()
    
    logger.info("训练完成!")
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="starVLA/config/training/geosuper.yaml",
        help="配置文件路径"
    )
    parser.add_argument("--enable_curriculum", type=bool, default=True, help="启用课程学习")
    parser.add_argument("--enable_geometric_debug", type=bool, default=True, help="启用几何调试")
    parser.add_argument("--enable_attention_visualization", type=bool, default=False, help="启用注意力可视化")
    parser.add_argument("--log_dir", type=str, default="./logs", help="日志目录")
    parser.add_argument("--num_gpus", type=int, default=8, help="GPU数量")
    parser.add_argument("--mixed_precision", type=str, default="bfloat16", help="混合精度")
    
    args, clipargs = parser.parse_known_args()
    
    # 加载配置
    cfg = OmegaConf.load(args.config_yaml)
    
    # 合并命令行参数
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    
    # 添加额外的配置
    cfg.enable_curriculum = args.enable_curriculum
    cfg.enable_geometric_debug = args.enable_geometric_debug
    cfg.enable_attention_visualization = args.enable_attention_visualization
    cfg.hardware.num_gpus = args.num_gpus
    cfg.trainer.mixed_precision_dtype = args.mixed_precision
    
    # 调试模式
    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 等待调试器连接端口 10092...")
        debugpy.wait_for_client()
    
    main(cfg)