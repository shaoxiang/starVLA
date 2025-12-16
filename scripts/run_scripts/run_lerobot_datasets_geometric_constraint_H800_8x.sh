#!/bin/bash
# QwenSuper-GeometricConstraint 训练脚本（H800 8卡）- 已修复多阶段切换和检查点路径问题
# 核心改进：正确处理分布式训练环境清理、进程同步和检查点文件路径

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
run_id=$(date +%m%d)_geoconstraint_H800_8x_fixed
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

# ==================== 辅助函数：等待进程结束 ====================
wait_for_processes() {
    local process_name="$1"
    local timeout=${2:-300}  # 默认5分钟超时
    
    echo "等待 ${process_name} 进程结束..."
    local count=0
    while pgrep -f "${process_name}" > /dev/null; do
        sleep 5
        count=$((count + 5))
        if [ $count -gt $timeout ]; then
            echo "⚠️ 超时等待 ${process_name}，强制终止相关进程"
            pkill -f "${process_name}"
            break
        fi
    done
    echo "${process_name} 进程已结束"
}

# ==================== 辅助函数：清理分布式环境 ====================
cleanup_distributed_env() {
    echo "正在清理分布式训练环境..."
    
    # 终止所有相关的Python和Accelerate进程
    pkill -f "accelerate.launch" 2>/dev/null || true
    pkill -f "train_geometric_constraint" 2>/dev/null || true
    pkill -f "torchrun" 2>/dev/null || true
    
    # 清理CUDA缓存
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    
    # 等待一段时间让系统清理资源
    sleep 10
    
    echo "分布式环境清理完成"
}

# ==================== 辅助函数：获取最新检查点文件 ====================
get_latest_checkpoint_file() {
    local checkpoint_dir="$1"
    local pattern="$2"
    
    if [ ! -d "$checkpoint_dir" ]; then
        echo ""
        return
    fi
    
    # 首先尝试找到 .pt 或 .pth 文件
    local checkpoint_file=$(find "$checkpoint_dir" -name "*.pt" -o -name "*.pth" | sort -V | tail -1)
    
    # 如果没找到，则查找包含 'model' 或 'checkpoint' 的子目录中的文件
    if [ -z "$checkpoint_file" ]; then
        for subdir in $(find "$checkpoint_dir" -maxdepth 1 -type d | tail -n +2 | sort); do
            checkpoint_file=$(find "$subdir" -name "*.pt" -o -name "*.pth" | sort -V | tail -1)
            if [ -n "$checkpoint_file" ]; then
                break
            fi
        done
    fi
    
    # 如果还是没找到，尝试查找最新的子目录
    if [ -z "$checkpoint_file" ]; then
        local latest_subdir=$(ls -td "$checkpoint_dir"/*/ 2>/dev/null | head -1)
        if [ -n "$latest_subdir" ]; then
            checkpoint_file=$(find "$latest_subdir" -name "*.pt" -o -name "*.pth" | sort -V | tail -1)
        fi
    fi
    
    echo "$checkpoint_file"
}

# ==================== 阶段0：基础适应训练 ====================
stage=0
stage_dir="${output_dir}/stage_${stage}"
mkdir -p ${stage_dir}
log_dir="${stage_dir}/logs"
mkdir -p ${log_dir}

echo "========================================"
echo "阶段 ${stage}: 基础动作预测训练"
echo "目标: 让DiT学会基本的动作生成"
echo "冻结: VLM, DINO, MapAnything"
echo "约束权重: 0.0"
echo "输出: ${stage_dir}"
echo "========================================"

# 设置阶段0的主进程端口
MAIN_PORT=29500

# 启动阶段0训练
{
    accelerate launch \
      --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
      --num_processes 8 \
      --main_process_port ${MAIN_PORT} \
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
      --debug False
} 2>&1 | tee "${stage_dir}/logs/train_stage${stage}_${timestamp}.log"

# 检查阶段0是否成功
TRAIN_EXIT_CODE=$?
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "❌ 阶段 ${stage} 训练失败！退出码: ${TRAIN_EXIT_CODE}"
    echo "查看日志: ${stage_dir}/logs/"
    cleanup_distributed_env
    exit $TRAIN_EXIT_CODE
fi

echo "✓ 阶段 ${stage} 训练完成！"

# 等待所有相关进程结束
wait_for_processes "train_geometric_constraint"

# 清理分布式环境
cleanup_distributed_env

# 获取阶段0的最后检查点文件（修复：获取具体文件而不是目录）
stage0_checkpoint=$(get_latest_checkpoint_file "${stage_dir}/checkpoints")
if [ -n "${stage0_checkpoint}" ] && [ -f "${stage0_checkpoint}" ]; then
    echo "阶段0检查点文件: ${stage0_checkpoint}"
else
    echo "❌ 未找到有效的阶段0检查点文件"
    echo "检查目录: ${stage_dir}/checkpoints"
    ls -la "${stage_dir}/checkpoints" 2>/dev/null || echo "检查点目录不存在"
    exit 1
fi

# ==================== 等待系统资源稳定 ====================
echo "等待系统资源稳定..."
sleep 30

# ==================== 阶段1：联合优化训练 ====================
stage=1
stage_dir="${output_dir}/stage_${stage}"
mkdir -p ${stage_dir}
log_dir="${stage_dir}/logs"
mkdir -p ${log_dir}

echo "========================================"
echo "阶段 ${stage}: 联合优化训练"
echo "目标: 几何约束与动作预测协同优化"
echo "解冻: 所有模块"
echo "约束权重: 0.2"
echo "使用检查点: ${stage0_checkpoint}"
echo "输出: ${stage_dir}"
echo "========================================"

# 设置阶段1的主进程端口（避免冲突）
MAIN_PORT=29501

# 从阶段0恢复训练
{
    accelerate launch \
      --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
      --num_processes 8 \
      --main_process_port ${MAIN_PORT} \
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
      --debug False
} 2>&1 | tee "${stage_dir}/logs/train_stage${stage}_${timestamp}.log"

# 检查阶段1是否成功
TRAIN_EXIT_CODE=$?
if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "❌ 阶段 ${stage} 训练失败！退出码: ${TRAIN_EXIT_CODE}"
    echo "查看日志: ${stage_dir}/logs/"
    cleanup_distributed_env
    exit $TRAIN_EXIT_CODE
fi

echo "✓ 阶段 ${stage} 训练完成！"

# 等待所有相关进程结束
wait_for_processes "train_geometric_constraint"

# ==================== 最终模型整理 ====================
final_dir="${output_dir}/final_model"
mkdir -p ${final_dir}

# 获取阶段1的最后检查点文件
final_checkpoint=$(get_latest_checkpoint_file "${stage_dir}/checkpoints")
if [ -n "${final_checkpoint}" ] && [ -f "${final_checkpoint}" ]; then
    cp "${final_checkpoint}" "${final_dir}/model.pt"
    echo "✓ 最终模型已保存: ${final_dir}/model.pt"
else
    echo "❌ 寻找最终检查点文件..."
    final_checkpoint=$(find "${stage_dir}" -name "*.pt" -o -name "*.pth" | sort -V | tail -1)
    if [ -n "${final_checkpoint}" ] && [ -f "${final_checkpoint}" ]; then
        cp "${final_checkpoint}" "${final_dir}/model.pt"
        echo "✓ 备用最终模型已保存: ${final_dir}/model.pt"
    else
        echo "❌ 仍未找到有效的最终检查点文件"
        find "${stage_dir}" -name "*" -type f | grep -E "\.(pt|pth)$" || echo "没有找到 .pt 或 .pth 文件"
        exit 1
    fi
fi

# 清理最终的分布式环境
cleanup_distributed_env

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
final_model_path = Path(base_dir) / 'final_model' / 'model.pt'
if final_model_path.exists():
    print(f'  - 最终模型: {final_model_path}')
else:
    print('  - 最终模型: 未找到')

stage0_log = Path(base_dir) / 'stage_0' / 'logs'
stage1_log = Path(base_dir) / 'stage_1' / 'logs'
print(f'  - 阶段0日志: {stage0_log}')
print(f'  - 阶段1日志: {stage1_log}')
print(f'  - 阶段0检查点: {\"${stage0_checkpoint}\"}')
print(f'  - 最终检查点: {\"${final_checkpoint}\"}')
print('========================================')
" 2>&1 | tee "${output_dir}/training_summary.txt"

echo "========================================"
echo "🎉 训练完成！"
echo "输出目录: ${output_dir}"
echo "训练总结: ${output_dir}/training_summary.txt"
echo "WandB项目: geoconstraint-vla"
echo "========================================"