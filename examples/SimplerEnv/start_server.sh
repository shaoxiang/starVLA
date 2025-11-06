

check_pt=/data/models/starVLA/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt
sim_python=/data/conda/simpler_env/bin/python
sim_python=/data/conda/starVLA/bin/python
port=5678
# DEBUG=true

CUDA_VISIBLE_DEVICES=2 ${sim_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16