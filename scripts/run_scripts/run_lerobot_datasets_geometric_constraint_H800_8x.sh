#!/bin/bash
# QwenSuper-GeometricConstraint 优化版训练脚本
# 功能：支持全流程训练，也支持指定模型路径从 Stage 1 单独启动

# ==================== 1. 核心控制面板 (修改这里即可) ====================

# [控制] 起始阶段: 0 = 从头开始(Stage0->Stage1); 1 = 跳过Stage0, 直接跑Stage1
START_STAGE=1

# [控制] 手动指定 Stage 0 模型路径
# 注意：仅当 START_STAGE=1 时，此路径生效。
# 如果你从 Stage 0 开始跑，脚本会自动处理路径，忽略此变量。
PRETRAINED_PATH="/public/home/vlabadmin/dataset/starVLA/qwen_geoconstraint/1216_geoconstraint_H800_8x_fixed_20251216_165821/stage_0/1216_geoconstraint_H800_8x_fixed_stage0_20251216_165821/final_model/model.pt"

# ==================== 2. 基础环境配置 ====================
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export WANDB_MODE=disabled

# 项目路径配置
Framework_name="QwenSuper-GeometricConstraint"
base_vlm="/public/home/vlabadmin/dataset/Qwen3-VL-4B-Instruct"
oxe_data_root="/public/home/vlabadmin/dataset/OXE_LEROBOT_DATASET"
run_root_dir="/public/home/vlabadmin/dataset/starVLA/qwen_geoconstraint"

# 自动生成输出目录名
timestamp=$(date +%Y%m%d_%H%M%S)
run_id="opt_H800_8x"
output_dir="${run_root_dir}/${run_id}_${timestamp}"

# 打印配置信息
echo "========================================"
echo "🚀 训练启动"
echo "起始阶段: Stage ${START_STAGE}"
if [ "${START_STAGE}" -eq 1 ]; then
    echo "加载手动指定模型: ${PRETRAINED_PATH}"
fi
echo "输出总目录: ${output_dir}"
echo "========================================"

# 创建日志目录
mkdir -p "${output_dir}"

# 辅助函数：清理环境
cleanup_env() {
    echo "🧹 清理残留进程..."
    pkill -f "train_geometric_constraint" 2>/dev/null || true
    pkill -f "accelerate.launch" 2>/dev/null || true
    sleep 3
}

# ==================== 3. Stage 0 训练模块 ====================
if [ "${START_STAGE}" -le 0 ]; then
    stage_dir="${output_dir}/stage_0"
    mkdir -p "${stage_dir}/logs"
    
    echo -e "\n>>> 进入 Stage 0: 基础动作预测训练"
    
    # 清理环境确保端口可用
    cleanup_env

    accelerate launch \
      --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
      --num_processes 8 \
      --main_process_port 29500 \
      starVLA/training/train_geometric_constraint.py \
      --config_yaml ./starVLA/config/training/geometric_constraint.yaml \
      --framework.name ${Framework_name} \
      --framework.qwenvl.base_vlm ${base_vlm} \
      --framework.action_model.action_model_type "DiT-B" \
      --framework.dino.dino_backbone "dinov3_vits16" \
      --framework.map_anything.model_repo_id "facebook/map-anything" \
      --datasets.vla_data.data_root_dir ${oxe_data_root} \
      --datasets.vla_data.data_mix "bridge_rt_1" \
      --datasets.vla_data.per_device_batch_size 16 \
      --trainer.curriculum_stage 0 \
      --trainer.constraint_weight 0.0 \
      --trainer.max_train_steps 40000 \
      --trainer.save_interval 10000 \
      --trainer.logging_frequency 50 \
      --trainer.learning_rate.main 1e-5 \
      --trainer.gradient_accumulation_steps 2 \
      --run_root_dir ${stage_dir} \
      --run_id "stage0_${timestamp}" \
      --wandb_project geoconstraint-vla \
      --debug False 2>&1 | tee "${stage_dir}/logs/train_stage0.log"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "❌ Stage 0 训练失败，脚本终止"
        exit 1
    fi

    echo "✅ Stage 0 完成"
    
    # 自动查找 Stage 0 产出的模型（修复路径逻辑）
    # 使用 find 强力查找 stage_dir 下所有的 final_model/model.pt
    FOUND_MODEL=$(find "${stage_dir}" -name "model.pt" | grep "final_model" | head -n 1)
    
    if [ -f "$FOUND_MODEL" ]; then
        PRETRAINED_PATH="$FOUND_MODEL"
        echo "🔗 自动锁定 Stage 0 模型: ${PRETRAINED_PATH}"
    else
        echo "❌ 严重错误: 无法在 ${stage_dir} 下找到生成的模型文件"
        exit 1
    fi
fi

# ==================== 4. Stage 1 训练模块 ====================
if [ "${START_STAGE}" -le 1 ]; then
    stage_dir="${output_dir}/stage_1"
    mkdir -p "${stage_dir}/logs"

    echo -e "\n>>> 进入 Stage 1: 联合优化训练"
    
    # 检查模型是否存在
    if [ ! -f "${PRETRAINED_PATH}" ]; then
        echo "❌ 错误: Stage 1 需要加载的模型文件不存在!"
        echo "路径: ${PRETRAINED_PATH}"
        exit 1
    fi

    # 清理环境 (Stage 0 刚跑完或手动启动都需要清理)
    cleanup_env

    accelerate launch \
      --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
      --num_processes 8 \
      --main_process_port 29501 \
      starVLA/training/train_geometric_constraint.py \
      --config_yaml ./starVLA/config/training/geometric_constraint.yaml \
      --framework.name ${Framework_name} \
      --framework.qwenvl.base_vlm ${base_vlm} \
      --framework.action_model.action_model_type "DiT-B" \
      --framework.dino.dino_backbone "dinov3_vits16" \
      --framework.map_anything.model_repo_id "facebook/map-anything" \
      --datasets.vla_data.data_root_dir ${oxe_data_root} \
      --datasets.vla_data.data_mix "bridge_rt_1" \
      --datasets.vla_data.per_device_batch_size 16 \
      --trainer.curriculum_stage 1 \
      --trainer.constraint_weight 0.2 \
      --trainer.max_train_steps 40000 \
      --trainer.save_interval 10000 \
      --trainer.logging_frequency 50 \
      --trainer.learning_rate.main 5e-6 \
      --trainer.learning_rate.constraint_head 1e-4 \
      --trainer.num_warmup_steps 500 \
      --trainer.pretrained_checkpoint "${PRETRAINED_PATH}" \
      --trainer.reload_modules "action_model,geom_constraint_head" \
      --trainer.gradient_accumulation_steps 2 \
      --run_root_dir ${stage_dir} \
      --run_id "stage1_${timestamp}" \
      --wandb_project geoconstraint-vla \
      --debug False 2>&1 | tee "${stage_dir}/logs/train_stage1.log"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "❌ Stage 1 训练失败"
        exit 1
    fi
    
    echo "🎉 所有训练任务完成！"
    echo "最终结果位于: ${stage_dir}"
fi