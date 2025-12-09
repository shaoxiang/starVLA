#!/bin/bash
# Qwen-GeoSuper 训练脚本（适配 H800 8卡）- 支持自动课程学习
# 文件名：run_lerobot_datasets_geoSuper_H800_8x.sh

export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200
export WANDB_MODE=disabled  # 实际训练时启用wandb

# ==================== 基础配置 ====================
Framework_name="Qwen-GeoSuper"
base_vlm="/public/home/vlabadmin/dataset/Qwen3-VL-4B-Instruct"

# 模型配置
DIT_TYPE="DiT-B"
MAP_MODEL="facebook/map-anything"
dino_backbone="dinov3_vits16"

# 课程学习阶段配置（按顺序执行）
# 格式：阶段序号:最大步数:冻结模块:损失权重(动作:可供性:物理约束)
STAGES_CONFIG=(
    "0:40000:qwen_vl_interface.model.model.visual,dino_encoder,map_encoder,affordance_module,ego_geom_transform:1.0:0.0:0.0"
    "1:40000:qwen_vl_interface.model.model.visual,dino_encoder:1.0:0.1:0.05"
    "2:40000::1.0:0.15:0.1"  # 空字符串表示无冻结模块
)

# 数据配置
llavadata="asv2_conversation_en,asv2_detailed_description_en"
oxe_data_root="/public/home/vlabadmin/dataset/OXE_LEROBOT_DATASET"
data_mix="bridge_rt_1"

# 输出配置
run_root_dir="/public/home/vlabadmin/dataset/starVLA/qwen_geosuper/Checkpoints"
base_run_id="$(date +%m%d)_geosuper_curriculum"
timestamp=$(date +%Y%m%d_%H%M%S)
base_output_dir="${run_root_dir}/${base_run_id}_${timestamp}"
mkdir -p ${base_output_dir}

# 复制脚本到输出目录（保存原始配置）
cp "$0" "${base_output_dir}/"

# ==================== 课程学习主循环 ====================
echo "========================================"
echo "开始 Qwen-GeoSuper 课程学习训练"
echo "总阶段数: ${#STAGES_CONFIG[@]}"
echo "基础输出目录: ${base_output_dir}"
echo "========================================"

# 记录累计训练步数
total_accumulated_steps=0

# 遍历所有阶段
for stage_config in "${STAGES_CONFIG[@]}"; do
    # 解析阶段配置
    IFS=':' read -r stage max_steps freeze_modules action_weight affordance_weight physical_weight <<< "$stage_config"
    
    # 阶段输出目录
    stage_output_dir="${base_output_dir}/stage_${stage}"
    mkdir -p ${stage_output_dir}
    log_dir="${stage_output_dir}/logs"
    mkdir -p ${log_dir}
    
    echo "========================================"
    echo "进入阶段 ${stage}"
    echo "阶段最大步数: ${max_steps}"
    echo "冻结模块: ${freeze_modules:-无}"
    echo "损失权重 - 动作: ${action_weight}, 可供性: ${affordance_weight}, 物理约束: ${physical_weight}"
    echo "阶段输出目录: ${stage_output_dir}"
    echo "========================================"

    # 启动训练（核心修改：移除可能导致stages类型冲突的参数）
    accelerate launch \
      --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
      --num_processes 8 \
      --main_process_port $((29500 + stage)) \
      starVLA/training/train_geosuper.py \
      --config_yaml ./starVLA/config/training/geosuper.yaml \
      --framework.name ${Framework_name} \
      --framework.qwenvl.base_vlm ${base_vlm} \
      --framework.action_model.action_model_type ${DIT_TYPE} \
      --framework.dino.dino_backbone ${dino_backbone} \
      --framework.map_anything.model_repo_id ${MAP_MODEL} \
      --datasets.vla_data.data_root_dir ${oxe_data_root} \
      --datasets.vla_data.data_mix ${data_mix} \
      --datasets.vla_data.per_device_batch_size 16 \
      --trainer.training_strategy "curriculum" \
      --trainer.curriculum_stage ${stage} \
      --trainer.freeze_modules "${freeze_modules}" \
      --trainer.max_train_steps ${max_steps} \
      --trainer.save_interval 10000 \
      --trainer.logging_frequency 10 \
      --trainer.eval_interval 100 \
      --trainer.learning_rate.base 4e-5 \
      --trainer.loss_weights.action_loss_weight ${action_weight} \
      --trainer.loss_weights.affordance_loss_weight ${affordance_weight} \
      --trainer.loss_weights.physical_constraint_weight ${physical_weight} \
      --run_root_dir ${stage_output_dir} \
      --run_id "${base_run_id}_stage${stage}_${timestamp}" \
      --wandb_project starVLA_geosuper \
      --wandb_entity jinhuiye \
      --log_dir ${log_dir} \
      --enable_curriculum true \
      --enable_geometric_debug true \
      --enable_attention_visualization true \
      --num_gpus 8 \
      --mixed_precision bfloat16 \
      --gradient_checkpointing true \
      --gradient_accumulation_steps 2 \
      2>&1 | tee "${log_dir}/train_stage${stage}_${timestamp}.log"

    # 检查训练是否成功完成
    if [ $? -ne 0 ]; then
        echo "阶段 ${stage} 训练失败！请检查日志: ${log_dir}/train_stage${stage}_${timestamp}.log"
        exit 1
    fi

    # 更新累计步数
    total_accumulated_steps=$((total_accumulated_steps + max_steps))
    echo "阶段 ${stage} 训练完成！累计训练步数: ${total_accumulated_steps}"

    # 保存阶段 checkpoint 路径，用于下一阶段加载（如果需要）
    last_checkpoint=$(ls -td ${stage_output_dir}/checkpoints/* | head -1)
    if [ -n "${last_checkpoint}" ]; then
        echo "当前阶段最后 checkpoint: ${last_checkpoint}"
        export PRETRAINED_CHECKPOINT=${last_checkpoint}
    else
        echo "警告：未找到阶段 ${stage} 的 checkpoint！"
        export PRETRAINED_CHECKPOINT=""
    fi
done

# ==================== 训练总结 ====================
echo "========================================"
echo "所有课程学习阶段训练完成！"
echo "总累计训练步数: ${total_accumulated_steps}"
echo "完整输出目录: ${base_output_dir}"
echo "========================================"

# 生成整体训练总结
python -c "
import json
import pandas as pd
import os
from pathlib import Path

base_dir = '${base_output_dir}'
all_data = []

# 收集所有阶段的日志
for stage_dir in Path(base_dir).glob('stage_*'):
    summary_path = stage_dir / 'summary.jsonl'
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                data['stage'] = stage_dir.name.split('_')[-1]  # 记录数据所属阶段
                all_data.append(data)

if all_data:
    df = pd.DataFrame(all_data)
    print('整体训练统计:')
    print(f'总阶段数: {len(df["stage"].unique())}')
    print(f'总记录步数: {len(df)}')
    print(f'最终总损失: {df["total_loss"].iloc[-1] if "total_loss" in df.columns else "N/A"}')
    
    # 保存总结到文件
    summary_file = os.path.join(base_dir, 'overall_training_summary.txt')
    with open(summary_file, 'w') as f:
        f.write('Qwen-GeoSuper 课程学习训练总结\n')
        f.write(f'总阶段数: {len(df["stage"].unique())}\n')
        f.write(f'总记录步数: {len(df)}\n')
        f.write(f'最终总损失: {df["total_loss"].iloc[-1] if "total_loss" in df.columns else "N/A"}\n')
    print(f'总结已保存至: {summary_file}')
else:
    print('未找到训练日志数据，无法生成总结')
" 2>&1 | tee "${base_output_dir}/overall_training_summary.txt"