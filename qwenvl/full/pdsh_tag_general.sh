#!/bin/bash

# 多节点分布式训练启动脚本
export PDSH_RCMD_TYPE=ssh

# 解析节点IP列表
node_ips=($(echo ${NODE_IP_LIST} | sed 's/:8//g'))

# 设置主节点
export MASTER_ADDR=${node_ips[0]}

# 集群范围选择统一可用端口（随机起点 + 跨节点实际 bind 测试）
get_cluster_free_port() {
    local start_port=${1}
    if [ -z "$start_port" ]; then
        start_port=$((36000 + RANDOM % 20000))
    fi
    local port=$start_port
    while true; do
        # 仅在主节点检测端口是否被占用（连接式检测：若能连接说明已被占用）
        # 分布式只在 MASTER 上监听该端口，因此只需确保 MASTER 节点空闲
        local check_output
        check_output=$(pdsh -w "$MASTER_ADDR" "python3 -c \"import socket as s; p=$port; so=s.socket(s.AF_INET, s.SOCK_STREAM); so.settimeout(0.5); r=so.connect_ex(('127.0.0.1', p)); so.close(); print('used' if r==0 else 'free')\"" 2>/dev/null)
        if ! echo "$check_output" | grep -q used; then
            echo $port
            return 0
        fi
        port=$((port+1))
        if [ $port -gt 65000 ]; then
            port=36000
        fi
    done
}

export MASTER_PORT=$(get_cluster_free_port 36000)
echo "[pdsh.sh] Using MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# # 在所有节点上设置环境
# pdsh -w $(IFS=,; echo "${node_ips[*]}") "source /apdcephfs_gy2/share_302507476/xiaodayang/miniconda3/bin/activate ; conda activate qwenvl"
pdsh -w $(IFS=,; echo "${node_ips[*]}") "pip install transformers==4.57.1; pip install accelerate==1.11.0"

# 创建配置文件目录
mkdir -p /apdcephfs_gy2/share_302507476/xiaodayang/SpatialLogic/accelerate_config_machine

# 机器数量（从节点列表推断）
NUM_MACHINES=${#node_ips[@]}

# 为每台机器创建配置文件
for i in "${!node_ips[@]}"; do
    config_file="/apdcephfs_gy2/share_302507476/xiaodayang/SpatialLogic/accelerate_config_machine/${i}.yaml"
    
    cat > $config_file << EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: 'all'
machine_rank: $i
main_training_function: main
main_process_port: $MASTER_PORT
mixed_precision: bf16
num_machines: $NUM_MACHINES
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
done

# 创建 DeepSpeed 配置文件
ds_config_file="/apdcephfs_gy2/share_302507476/xiaodayang/SpatialLogic/accelerate_config_machine/ds_config.json"

cat > $ds_config_file << 'EOF'
{
    "fp16": {
        "enabled": false
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 1,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": "auto"
        }
    }
}
EOF

# 在每台机器上并行启动训练
for i in "${!node_ips[@]}"; do
    ip=${node_ips[$i]}
    config_file="/apdcephfs_gy2/share_302507476/xiaodayang/SpatialLogic/accelerate_config_machine/${i}.yaml"
    
    pdsh -w $ip "
        export MASTER_ADDR=$MASTER_ADDR && \
        export MASTER_PORT=$MASTER_PORT && \
        accelerate launch --config_file $config_file --main_process_port $MASTER_PORT \
            /apdcephfs_nj8/share_301739632/xiaodayang/code/SpatialLogic/full/train_tag_general.py \
            --accelerate \
            --deepspeed $ds_config_file \
            --model_path /apdcephfs_nj8/share_301739632/xiaodayang/data/SpatialLogic/LOVE-Agibot-Beta/checkpoint-60000 \
            --processor_path /apdcephfs_gy2/share_302507476/xiaodayang/SpatialLogic/ckpt/qwenvl/original/Qwen2.5-VL-7B-Instruct \
            --tag_files /apdcephfs_nj8/share_301739632/xiaodayang/data/SpatialLogic/LOVE-Agibot-Beta/General/data/train_41936.json \
            --dataloader_workers 8" &
done

# 等待所有训练进程
wait
#/apdcephfs_nj8/share_301739632/xiaodayang/data/SpatialLogic/LOVE-Agibot-Beta/tag_json/long_tag_step_total.json
#/apdcephfs_gy2/share_302507476/xiaodayang/SpatialLogic/data/json/Exp/tag10fin.json