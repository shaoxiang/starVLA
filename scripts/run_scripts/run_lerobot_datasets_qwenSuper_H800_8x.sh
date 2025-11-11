export NCCL_SOCKET_IFNAME=eth0  # 或 ib0（Infiniband）

# 只保留必要的调试配置
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200

Framework_name=Qwen-Super
base_vlm=/public/home/vlabadmin/dataset/Qwen3-VL-4B-Instruct

DIT_TYPE="DiT-B"
freeze_module_list="qwen_vl_interface.model.model.visual,dino_encoder" # just for fast debug, sota is under fully FT, i.g., freeze_module_list=""

llavadata="asv2_conversation_en,asv2_detailed_description_en"
oxe_data_root=/public/home/vlabadmin/dataset/OXE_LEROBOT_DATASET
data_mix=bridge_rt_1

dino_backbone=dinov2_vits14

run_root_dir=/public/home/vlabadmin/dataset/starVLA/qwen_super/Checkpoints
run_id=1111_supervla_qwen_gr00t_dinov3

export action_input_dim=2
export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/supervla_cotrain_oxe.yaml \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.action_hidden_dim ${action_input_dim} \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --framework.dino.dino_backbone ${dino_backbone} \
  --datasets.vla_data.data_root_dir ${oxe_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --trainer.learning_rate.base 4e-5 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA \
  --wandb_entity jinhuiye \
  # --is_debug True


