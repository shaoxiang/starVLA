#!/bin/bash
# Qwen-GeoSuper 训练脚本（适配 H800 8卡）
# 文件名：run_lerobot_datasets_geoSuper_H800_8x.sh

export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200
export WANDB_MODE=disabled  # 调试时禁用wandb

# ==================== 基础配置 ====================
Framework_name="Qwen-GeoSuper"
base_vlm="/public/home/vlabadmin/dataset/Qwen3-VL-4B-Instruct"

# 模型配置
DIT_TYPE="DiT-B"
MAP_MODEL="facebook/map-anything-apache"  # 使用Apache许可的模型
dino_backbone="dinov3_vits16"  # 或 dinov2_vits14、dinov3_vits16

# 训练策略配置
TRAIN_STRATEGY="curriculum"  # curriculum, end2end, selective
CURRICULUM_STAGE=0  # 初始阶段：0, 1, 2
FREEZE_MODULES_STAGE0="qwen_vl_interface.model.model.visual,dino_encoder,map_encoder,affordance_module,ego_geom_transform"
FREEZE_MODULES_STAGE1="qwen_vl_interface.model.model.visual,dino_encoder"
FREEZE_MODULES_STAGE2=""  # 第三阶段全部解冻

# 数据配置
llavadata="asv2_conversation_en,asv2_detailed_description_en"  # VLM数据
oxe_data_root="/public/home/vlabadmin/dataset/OXE_LEROBOT_DATASET"
data_mix="bridge_rt_1"

# 输出配置
run_root_dir="/public/home/vlabadmin/dataset/starVLA/qwen_geosuper/Checkpoints"
run_id="$(date +%m%d)_geosuper_${TRAIN_STRATEGY}_stage${CURRICULUM_STAGE}"
timestamp=$(date +%Y%m%d_%H%M%S)

output_dir="${run_root_dir}/${run_id}_${timestamp}"
mkdir -p ${output_dir}

# 复制脚本到输出目录
cp "$0" "${output_dir}/"

# 根据训练阶段选择冻结模块
case $CURRICULUM_STAGE in
    0)
        freeze_module_list=$FREEZE_MODULES_STAGE0
        echo "使用阶段0冻结模块: $freeze_module_list"
        ;;
    1)
        freeze_module_list=$FREEZE_MODULES_STAGE1
        echo "使用阶段1冻结模块: $freeze_module_list"
        ;;
    2)
        freeze_module_list=$FREEZE_MODULES_STAGE2
        echo "使用阶段2冻结模块: $freeze_module_list"
        ;;
    *)
        freeze_module_list=""
        echo "使用端到端训练，无冻结模块"
        ;;
esac

# ==================== 训练命令 ====================
echo "========================================"
echo "开始训练 Qwen-GeoSuper"
echo "框架: $Framework_name"
echo "VLM: $base_vlm"
echo "DINO: $dino_backbone"
echo "MapAnything: $MAP_MODEL"
echo "训练策略: $TRAIN_STRATEGY (阶段$CURRICULUM_STAGE)"
echo "冻结模块: $freeze_module_list"
echo "数据混合: $data_mix"
echo "输出目录: $output_dir"
echo "========================================"

# 创建日志目录
log_dir="${output_dir}/logs"
mkdir -p ${log_dir}

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  --main_process_port 29500 \
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
  --trainer.training_strategy ${TRAIN_STRATEGY} \
  --trainer.curriculum_stage ${CURRICULUM_STAGE} \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --trainer.learning_rate.base 4e-5 \
  --trainer.curriculum_learning.stages[${CURRICULUM_STAGE}].max_steps 100000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id}_${timestamp} \
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
  2>&1 | tee "${log_dir}/train_${timestamp}.log"

echo "训练完成！输出目录: $output_dir"

# 生成训练总结
if [ -f "${output_dir}/summary.jsonl" ]; then
    echo "生成训练总结..."
    python -c "
import json
import pandas as pd
with open('${output_dir}/summary.jsonl', 'r') as f:
    lines = f.readlines()
data = [json.loads(line) for line in lines]
df = pd.DataFrame(data)
print('训练统计:')
print(f'总步数: {len(df)}')
print(f'最终损失: {df[\"total_loss\"].iloc[-1] if \"total_loss\" in df.columns else \"N/A\"}')
print(f'动作损失: {df[\"action_loss\"].iloc[-1] if \"action_loss\" in df.columns else \"N/A\"}')
print(f'可供性损失: {df[\"affordance_loss\"].iloc[-1] if \"affordance_loss\" in df.columns else \"N/A\"}')
" 2>&1 | tee "${output_dir}/training_summary.txt"
fi