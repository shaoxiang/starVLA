#!/bin/bash
# QwenSuper-GeometricConstraint 训练脚本（H800 8卡）
# 核心特点：2阶段课程学习，自动切换

export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200
export WANDB_MODE=disabled  # 训练时启用wandb

# ==================== 基础配置 ====================
Framework_name="QwenSuper-GeometricConstraint"
base_vlm="/public/home/vlabadmin/dataset/Qwen3-VL-4B-Instruct"

# 模型配置
DIT_TYPE="DiT-B"
MAP_MODEL="facebook/map-anything"
dino_backbone="dinov3_vits16"

# 数据配置
oxe_data_root="/public/home/vlabadmin/dataset/OXE_LEROBOT_DATASET"
data_mix="bridge_rt_1"

# 输出配置
run_root_dir="/public/home/vlabadmin/dataset/starVLA/qwen_geoconstraint"
timestamp=$(date +%Y%m%d_%H%M%S)
run_id=$(date +%m%d)_geoconstraint_H800_8x
output_dir="${run_root_dir}/${run_id}_${timestamp}"
mkdir -p ${output_dir}/logs

# 复制脚本（保存配置）
cp "$0" "${output_dir}/"

echo "========================================"
echo "QwenSuper-GeometricConstraint 训练启动"
echo "框架: ${Framework_name}"
echo "输出目录: ${output_dir}"
echo "数据: ${data_mix}"
echo "VLM: ${base_vlm}"
echo "========================================"

# ==================== 阶段0：基础适应训练 ====================
stage=0
stage_dir="${output_dir}/stage_${stage}"
mkdir -p ${stage_dir}
log_dir="${stage_dir}/logs"
mkdir -p ${log_dir}  # 确保日志目录存在

echo "========================================"
echo "阶段 ${stage}: 基础动作预测训练"
echo "目标: 让DiT学会基本的动作生成"
echo "冻结: VLM, DINO, MapAnything"
echo "约束权重: 0.0"
echo "输出: ${stage_dir}"
echo "========================================"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  --main_process_port 29500 \
  starVLA/training/train_geometric_constraint.py \
  --config_yaml ./starVLA/config/training/geometric_constraint.yaml \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --framework.dino.dino_backbone ${dino_backbone} \
  --framework.map_anything.model_repo_id ${MAP_MODEL} \
  --datasets.vla_data.data_root_dir ${oxe_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.curriculum_stage ${stage} \
  --trainer.constraint_weight 0.0 \
  --trainer.max_train_steps 40000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 50 \
  --trainer.learning_rate.main 1e-5 \
  --trainer.gradient_accumulation_steps 2 \
  --run_root_dir ${stage_dir} \
  --run_id "${run_id}_stage${stage}_${timestamp}" \
  --wandb_project geoconstraint-vla \
  --wandb_entity jinhuiye \
  --debug False \
  2>&1 | tee "${stage_dir}/logs/train_stage${stage}_${timestamp}.log"

# 检查阶段0是否成功
if [ $? -ne 0 ]; then
    echo "❌ 阶段 ${stage} 训练失败！查看日志: ${stage_dir}/logs/"
    exit 1
fi

echo "✓ 阶段 ${stage} 训练完成！"

# 获取阶段0的最后检查点
last_checkpoint=$(ls -td ${stage_dir}/checkpoints/*_model.pt | head -1)
if [ -n "${last_checkpoint}" ]; then
    stage0_checkpoint=${last_checkpoint}
    echo "阶段0检查点: ${stage0_checkpoint}"
else
    echo "❌ 未找到阶段0检查点"
    exit 1
fi

# ==================== 阶段1：联合优化训练 ====================
stage=1
stage_dir="${output_dir}/stage_${stage}"
mkdir -p ${stage_dir}
log_dir="${stage_dir}/logs"
mkdir -p ${log_dir}  # 确保日志目录存在

echo "========================================"
echo "阶段 ${stage}: 联合优化训练"
echo "目标: 几何约束与动作预测协同优化"
echo "解冻: 所有模块"
echo "约束权重: 0.2"
echo "输出: ${stage_dir}"
echo "========================================"

# 从阶段0恢复训练
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  --main_process_port 29501 \
  starVLA/training/train_geometric_constraint.py \
  --config_yaml ./starVLA/config/training/geometric_constraint.yaml \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --framework.dino.dino_backbone ${dino_backbone} \
  --framework.map_anything.model_repo_id ${MAP_MODEL} \
  --datasets.vla_data.data_root_dir ${oxe_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.curriculum_stage ${stage} \
  --trainer.constraint_weight 0.2 \
  --trainer.max_train_steps 40000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 50 \
  --trainer.learning_rate.main 5e-6 \
  --trainer.learning_rate.constraint_head 1e-4 \
  --trainer.num_warmup_steps 500 \
  --trainer.pretrained_checkpoint ${stage0_checkpoint} \
  --trainer.reload_modules "action_model,geom_constraint_head" \
  --trainer.gradient_accumulation_steps 2 \
  --run_root_dir ${stage_dir} \
  --run_id "${run_id}_stage${stage}_${timestamp}" \
  --wandb_project geoconstraint-vla \
  --wandb_entity jinhuiye \
  --debug False \
  2>&1 | tee "${stage_dir}/logs/train_stage${stage}_${timestamp}.log"

# 检查阶段1是否成功
if [ $? -ne 0 ]; then
    echo "❌ 阶段 ${stage} 训练失败！查看日志: ${stage_dir}/logs/"
    exit 1
fi

echo "✓ 阶段 ${stage} 训练完成！"

# ==================== 最终模型整理 ====================
final_dir="${output_dir}/final_model"
mkdir -p ${final_dir}

# 复制最后一个检查点
last_checkpoint_stage1=$(ls -td ${stage_dir}/checkpoints/*_model.pt | head -1)
if [ -n "${last_checkpoint_stage1}" ]; then
    cp ${last_checkpoint_stage1} ${final_dir}/model.pt
    echo "✓ 最终模型已保存: ${final_dir}/model.pt"
else
    echo "❌ 未找到最终检查点"
    exit 1
fi

# 生成训练总结
python -c "
import json
import pandas as pd
import os
from pathlib import Path

base_dir = '${output_dir}'
print('========================================')
print('QwenSuper-GeometricConstraint 训练完成')
print(f'输出目录: {base_dir}')
print('关键文件:')
print(f'  - 最终模型: {Path(base_dir) / 'final_model' / 'model.pt'}')
print(f'  - 配置: {Path(base_dir) / 'stage_1' / 'config.yaml'}')
print('========================================')
" 2>&1 | tee "${output_dir}/training_summary.txt"

echo "========================================"
echo "🎉 训练完成！"
echo "输出目录: ${output_dir}"
echo "训练总结: ${output_dir}/training_summary.txt"
echo "WandB项目: geoconstraint-vla"
echo "========================================"