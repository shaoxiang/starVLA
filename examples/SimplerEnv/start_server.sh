your_ckpt=/data/models/starVLA/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt
sim_python=/data/conda/starVLA/bin/python
port=5678

CUDA_VISIBLE_DEVICES=0 ${sim_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16