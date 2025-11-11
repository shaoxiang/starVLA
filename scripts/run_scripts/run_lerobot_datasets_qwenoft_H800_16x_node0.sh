# 设置多节点环境变量
export NCCL_SOCKET_IFNAME=eth0  # 使用eth0接口
export MASTER_ADDR=173.0.109.2  # 主节点IP
export MASTER_PORT=29500        # 主节点端口

# 最简单的配置 - 让NCCL自动检测
# unset NCCL_SOCKET_IFNAME  # 注释掉，使用上面明确的设置

# 只保留必要的调试配置
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

Framework_name=QwenOFT
base_vlm=/data/models/Qwen3-VL-4B-Instruct
base_vlm=/public/home/vlabadmin/dataset/Qwen3-VL-4B-Instruct  # 确保路径一致性

freeze_module_list='' # just for fast debug, sota is under fully FT, i.g., freeze_module_list=""

# freeze_module_list="qwen_vl_interface.model.model.visual,dino_encoder" # just for fast debug, sota is under fully FT, i.g., freeze_module_list=""

llavadata="asv2_conversation_en,asv2_detailed_description_en"

oxe_data_root=/data/dataset/OXE_LEROBOT_DATASET
oxe_data_root=/public/home/vlabadmin/dataset/OXE_LEROBOT_DATASET  # 确保路径一致性

data_mix=bridge_rt_1

run_root_dir=/data/models/starVLA/qwenoft/Checkpoints
run_root_dir=/public/home/vlabadmin/dataset/starVLA/qwenoft/Checkpoints1108  # 确保路径一致性

run_id=1106_starvla_qwenoft_oxe_h800_16x_2nodes  # 更新run_id以反映配置变化

export action_input_dim=2048
export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

# 使用多节点启动
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 16 \
  --num_machines 2 \
  --machine_rank 0 \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/starvla_cotrain_oxe.yaml \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.action_hidden_dim ${action_input_dim} \
  --datasets.vla_data.data_root_dir ${oxe_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 20000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --trainer.learning_rate.base 4e-5 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA \
  --wandb_entity jinhuiye \
  # --is_debug True