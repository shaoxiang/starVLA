cd .
port=6678
gpu_id=2
# export DEBUG=true
export star_vla_python=/data/conda/starVLA/bin/python
export PYTHONPATH=$(pwd):${PYTHONPATH}

MODEL_PATH=$1
if [ -z "$MODEL_PATH" ]; then
  echo "❌ MODEL_PATH not provided as the first argument, using default value"
fi
your_ckpt=${MODEL_PATH}

#### build output directory #####
ckpt_dir=$(dirname "${your_ckpt}")
ckpt_base=$(basename "${your_ckpt}")
ckpt_name="${ckpt_base%.*}"
output_server_dir="${ckpt_dir}/output_server"
mkdir -p "${output_server_dir}"
log_file="${output_server_dir}/${ckpt_name}_policy_server_${port}.log"


#### run server #####
CUDA_VISIBLE_DEVICES=${gpu_id} ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    2>&1 | tee "${log_file}"