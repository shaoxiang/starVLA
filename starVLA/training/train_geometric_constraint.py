# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
几何约束增强VLA模型训练脚本
核心特点：
  - 课程学习（2阶段：适应→联合优化）
  - 多loss监控（动作+几何约束）
  - 支持断点续训和模型分析
"""

import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import wandb
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
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups
from starVLA.dataloader import build_dataloader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 初始化加速器
deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
logger = get_logger(__name__)


class CurriculumManager:
    """简化版课程学习：2阶段策略"""
    
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        self.current_stage = getattr(cfg.trainer, 'curriculum_stage', 0)
        self.num_stages = 2  # 仅2阶段：适应→联合优化
        
        # 阶段配置
        self.stage_configs = [
            {
                "name": "stage0_adaptation",
                "freeze_modules": ["qwen_vl_interface.model.visual", "dino_encoder", "map_encoder"],
                "constraint_weight": 0.0,  # 第一阶段不激活约束
                "max_steps": cfg.trainer.stage0_steps,
            },
            {
                "name": "stage1_joint",
                "freeze_modules": [],  # 全部解冻
                "constraint_weight": cfg.trainer.constraint_weight,
                "max_steps": cfg.trainer.max_train_steps,
            },
        ]
        
        if accelerator.is_main_process:
            logger.info(f"课程学习初始化：{self.num_stages}阶段，当前阶段={self.current_stage}")
    
    def set_stage(self, stage: int):
        """切换到指定阶段"""
        if stage >= self.num_stages:
            return
        
        self.current_stage = stage
        config = self.stage_configs[stage]
        
        if accelerator.is_main_process:
            logger.info(f"切换到阶段 {stage}: {config['name']}")
            logger.info(f"  冻结模块: {config['freeze_modules']}")
            logger.info(f"  约束权重: {config['constraint_weight']}")
        
        # 应用模块冻结
        self._apply_freezing(config['freeze_modules'])
        
        # 更新模型约束权重
        if hasattr(self.model, 'constraint_weight'):
            self.model.constraint_weight = config['constraint_weight']
    
    def _apply_freezing(self, freeze_modules: List[str]):
        """冻结指定模块"""
        freeze_set = set([m.strip() for m in freeze_modules])
        
        # 默认所有参数可训练
        for param in self.model.parameters():
            param.requires_grad = True
        
        # 冻结指定模块
        for name, param in self.model.named_parameters():
            for module_name in freeze_set:
                if module_name in name:
                    param.requires_grad = False
                    if accelerator.is_main_process:
                        logger.debug(f"冻结参数: {name}")
        
        # 打印可训练参数统计
        if accelerator.is_main_process:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"可训练参数: {trainable_params/1e6:.2f}M/{total_params/1e6:.2f}M ({trainable_params/total_params*100:.1f}%)")
    
    def should_advance(self, current_step: int) -> bool:
        """判断是否需要进入下一阶段"""
        if self.current_stage >= self.num_stages - 1:
            return False
        
        next_stage_config = self.stage_configs[self.current_stage + 1]
        return current_step >= next_stage_config['max_steps']


class GeometricConstraintTrainer(TrainerUtils):
    """几何约束VLA训练器"""
    
    def __init__(self, cfg, model, train_dataloader, optimizer, lr_scheduler, accelerator):
        self.cfg = cfg
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        
        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        
        # 课程学习管理器
        self.curriculum = CurriculumManager(cfg, model)
        
        # 检查点目录
        self.checkpoint_dir = Path(cfg.output_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 监控配置
        self.enable_wandb = getattr(cfg, 'wandb_project', None) is not None
    
    def _calculate_total_batch_size(self):
        return (
            self.cfg.datasets.vla_data.per_device_batch_size *
            self.accelerator.num_processes *
            getattr(self.cfg.trainer, 'gradient_accumulation_steps', 1)
        )
    
    def prepare_training(self):
        """准备训练"""
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = getattr(self.cfg, 'seed', 42) + rank
        set_seed(seed)
        
        # 加载预训练权重
        if hasattr(self.cfg.trainer, "pretrained_checkpoint") and self.cfg.trainer.pretrained_checkpoint:
            pretrained_checkpoint = self.cfg.trainer.pretrained_checkpoint
            reload_modules = (
                self.cfg.trainer.reload_modules if hasattr(self.cfg.trainer, "reload_modules") else None
            )
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)

        # 冻结参数
        freeze_modules = (
            self.cfg.trainer.freeze_modules
            if (self.cfg and hasattr(self.cfg.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        
        # 初始化课程学习阶段
        self.curriculum.set_stage(self.cfg.trainer.curriculum_stage)
        
        # 分布式准备
        self.model, self.optimizer, self.train_dataloader = self.setup_distributed_training(
            self.accelerator,  # 必须是第一个参数
            self.model,
            self.optimizer,
            self.train_dataloader,
        )
        
        # 初始化WandB
        if self.enable_wandb and self.accelerator.is_main_process:
            wandb.init(
                name=self.cfg.run_id,
                dir=os.path.join(self.cfg.output_dir, "wandb"),
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity,
                group="vla-train",
                config=OmegaConf.to_container(self.cfg, resolve=True)
            )
        
        # 打印可训练参数
        self.print_trainable_parameters(self.model)
        
        if self.accelerator.is_main_process:
            logger.info("训练准备完成")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        if self.accelerator.is_main_process:
            logger.info(f"加载检查点: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 选择性加载模块
        reload_modules = getattr(self.cfg.trainer, 'reload_modules', None)
        if reload_modules:
            reload_set = set([m.strip() for m in reload_modules.split(',') if m.strip()])
            model_state_dict = self.model.state_dict()
            
            loaded_count = 0
            for name, param in checkpoint.items():
                for module_name in reload_set:
                    if module_name in name:
                        if name in model_state_dict:
                            model_state_dict[name] = param
                            loaded_count += 1
                            break
            
            self.model.load_state_dict(model_state_dict, strict=False)
            if self.accelerator.is_main_process:
                logger.info(f"选择性加载了 {loaded_count} 个参数")
        else:
            self.model.load_state_dict(checkpoint, strict=False)
    
    def _create_data_iterators(self):
        """创建数据迭代器"""
        self.train_iter = iter(self.train_dataloader)

    def _get_next_batch(self):
        """获取下一个批次（自动处理数据循环）"""
        try:
            batch = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_dataloader)
            batch = next(self.train_iter)
        return batch
    
    def train(self):
        """主训练循环"""
        if self.accelerator.is_main_process:
            logger.info("***** 开始训练 *****")
            logger.info(f"  总步数: {self.cfg.trainer.max_train_steps}")
            logger.info(f"  批次大小: {self.total_batch_size}")
        
        # 准备数据迭代器
        self._create_data_iterators()
        
        progress_bar = tqdm(
            range(self.cfg.trainer.max_train_steps),
            disable=not self.accelerator.is_local_main_process
        )
        
        while self.completed_steps < self.cfg.trainer.max_train_steps:
            # 获取数据批次
            t_start_data = time.perf_counter()
            batch = self._get_next_batch()
            t_end_data = time.perf_counter()
            
            # 训练步骤
            t_start_model = time.perf_counter()
            metrics = self._train_step(batch)
            t_end_model = time.perf_counter()
            
            # 更新进度
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1
            
            if self.accelerator.is_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )
            
            # 日志记录
            if self.completed_steps % self.cfg.trainer.logging_frequency == 0:
                # 添加学习率
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
                
                # 添加轮次信息
                metrics["epoch"] = round(self.completed_steps / len(self.train_dataloader), 2)
                
                self._log_metrics(metrics)
            
            # 保存检查点
            if self.completed_steps % self.cfg.trainer.save_interval == 0:
                self._save_checkpoint()
            
            # 评估（可选）
            if self.completed_steps % getattr(self.cfg.trainer, 'eval_interval', 1000) == 0:
                self._evaluate()
            
            # 课程学习阶段切换
            if self.curriculum.should_advance(self.completed_steps):
                self.curriculum.set_stage(self.curriculum.current_stage + 1)
            
            # 终止条件
            if self.completed_steps >= self.cfg.trainer.max_train_steps:
                break
        
        self._finalize_training()
    
    def _train_step(self, batch):
        """单个训练步骤"""
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            
            # 前向传播
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output_dict = self.model(batch)
                
                # 主损失（已包含约束加权）
                action_loss = output_dict["action_loss"]
                total_loss = output_dict["loss"]
            
            # 反向传播
            self.accelerator.backward(total_loss)
            
            # 梯度裁剪
            if getattr(self.cfg.trainer, 'gradient_clipping', None):
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.trainer.gradient_clipping
                )
            
            # 优化器步骤
            self.optimizer.step()
            self.lr_scheduler.step()
        
        # 收集指标
        metrics = {
            "total_loss": total_loss,
            "action_dit_loss": action_loss,
        }
        
        # 添加其他约束指标（如果存在）
        if "distance_constraint" in output_dict:
            metrics["distance_constraint"] = output_dict["distance_constraint"]
        if "workspace_constraint" in output_dict:
            metrics["workspace_constraint"] = output_dict["workspace_constraint"]
        if "orient_constraint" in output_dict:
            metrics["orient_constraint"] = output_dict["orient_constraint"]
        if "collision_constraint" in output_dict:
            metrics["collision_constraint"] = output_dict["collision_constraint"]
        if "debug_min_dist" in output_dict:
            metrics["debug_min_dist"] = output_dict["debug_min_dist"]
                
        metrics["curriculum_stage"] = self.curriculum.current_stage
        
        return metrics
    
    def _log_metrics(self, metrics: Dict):
        """记录指标"""
        if not self.accelerator.is_main_process:
            return
        
        # 打印日志
        log_str = f"步骤 {self.completed_steps}: total_loss={metrics['total_loss']:.4f}"
        if "action_dit_loss" in metrics:
            log_str += f", action_loss={metrics['action_dit_loss']:.4f}"
        if "distance_constraint" in metrics:
            log_str += f", distance_constraint={metrics['distance_constraint']:.4f}"
        if "workspace_constraint" in metrics:
            log_str += f", workspace_constraint={metrics['workspace_constraint']:.4f}"
        if "orient_constraint" in metrics:
            log_str += f", orient_constraint={metrics['orient_constraint']:.4f}"
        if "collision_constraint" in metrics:
            log_str += f", collision_constraint={metrics['collision_constraint']:.4f}"
        if "debug_min_dist" in metrics:
            log_str += f", min_dist={metrics['debug_min_dist']:.3f}m"
        log_str += f", lr={metrics['learning_rate']:.2e}"
        
        logger.info(log_str)
        
        # WandB记录
        if self.enable_wandb:
            wandb.log(metrics, step=self.completed_steps)
    
    def _save_checkpoint(self):
        """保存检查点"""
        if not self.accelerator.is_main_process:
            return
        
        checkpoint_path = self.checkpoint_dir / f"steps_{self.completed_steps}"
        
        # 保存模型状态
        model_state_dict = self.accelerator.get_state_dict(self.model)
        torch.save(model_state_dict, f"{checkpoint_path}_model.pt")
        
        # 保存优化器状态
        optimizer_state_dict = self.optimizer.state_dict()
        torch.save(optimizer_state_dict, f"{checkpoint_path}_optimizer.pt")
        
        # 保存元数据
        metadata = {
            'completed_steps': self.completed_steps,
            'curriculum_stage': self.curriculum.current_stage,
            'timestamp': time.time(),
        }
        torch.save(metadata, f"{checkpoint_path}_metadata.pt")
        
        logger.info(f"检查点已保存: {checkpoint_path}")
    
    def _evaluate(self):
        """快速评估（在训练集上采样）"""
        if not self.accelerator.is_main_process:
            return
        
        self.model.eval()
        try:
            batch = self._get_next_batch()
            with torch.no_grad():
                output = self.model(batch)
            
            # 记录评估指标
            eval_metrics = {
                "eval_total_loss": output["loss"].item(),
                "eval_action_loss": output["action_loss"].item(),
            }
            
            if "distance_constraint" in output:
                eval_metrics["eval_distance_constraint"] = output["distance_constraint"]
            if "workspace_constraint" in output:
                eval_metrics["eval_workspace_constraint"] = output["workspace_constraint"]
            if "orient_constraint" in output:
                eval_metrics["eval_orient_constraint"] = output["orient_constraint"]
            if "collision_constraint" in output:    
                eval_metrics["eval_collision_constraint"] = output["collision_constraint"]  
            if "debug_min_dist" in output:
                eval_metrics["eval_debug_min_dist"] = output["debug_min_dist"]
            if "debug_dist" in output:
                eval_metrics["eval_debug_dist"] = output["debug_dist"]

            logger.info(eval_metrics)
            
            if self.enable_wandb:
                wandb.log(eval_metrics, step=self.completed_steps)
        
        except Exception as e:
            logger.warning(f"评估失败: {e}")
        
        self.model.train()
    
    def _finalize_training(self):
        """训练结束"""
        if self.accelerator.is_main_process:
            final_dir = Path(self.cfg.output_dir) / "final_model"
            final_dir.mkdir(exist_ok=True)
            
            model_state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(model_state_dict, final_dir / "model.pt")
            
            logger.info(f"最终模型保存至: {final_dir}")
        
        if self.enable_wandb and self.accelerator.is_main_process:
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
    
    return output_dir


def build_model(cfg):
    """构建模型"""
    if accelerator.is_main_process:
        logger.info(f"构建模型: {cfg.framework.qwenvl.base_vlm}")
    model = build_framework(cfg)
    return model


def prepare_data(cfg):
    """准备数据加载器"""
    if accelerator.is_main_process:
        logger.info(f"加载数据: {cfg.datasets.vla_data.data_mix}")
    
    train_dataloader = build_dataloader(
        cfg=cfg,
        dataset_py=cfg.datasets.vla_data.dataset_py
    )
    
    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    
    return train_dataloader


def setup_optimizer_and_scheduler(model, cfg):
    """设置优化器和学习率调度器"""
    # 使用从原始trainer工具中提取的参数分组函数
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # 打印优化器组信息
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # 最小学习率
    )
    
    return optimizer, lr_scheduler


def main(cfg):
    """主函数"""
    if accelerator.is_main_process:
        logger.info("===== 几何约束VLA训练启动 =====")
    
    # 设置目录
    output_dir = setup_directories(cfg)
    
    # 构建模型
    model = build_model(cfg)
    
    # 准备数据
    train_dataloader = prepare_data(cfg)
    
    # 设置优化器
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model, cfg)
    
    # 创建训练器
    trainer = GeometricConstraintTrainer(
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
    
    if accelerator.is_main_process:
        logger.info("===== 训练完成 =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/geometric_constraint.yaml")
    args, clipargs = parser.parse_known_args()
    
    # 加载配置
    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.qwenvl.base_vlm = "/public/home/vlabadmin/dataset/Qwen3-VL-4B-Instruct"
    
    # 合并命令行参数
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    
    main(cfg)