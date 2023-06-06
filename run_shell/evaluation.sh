export CUDA_VISIBLE_DEVICES="1" # if using multi-gpu, put the multi gpu numbers, e.g., "2,3"
num_gpu=1 # gpu number
multi_gpu=false


export checkpoint_list=("0028000") # Put the iteration number of trained model
export output_dir="checkpoints/50/??" # Put the directory for evaluation

if $multi_gpu;then
    for t in ${checkpoint_list[@]}
    do
        python -m torch.distributed.launch --master_port 10093 --nproc_per_node=$num_gpu \
            tools/relation_test_net.py --config-file "${output_dir}/config.yml" \
            TEST.IMS_PER_BATCH 2 \
            OUTPUT_DIR ${output_dir} \
            MODEL.WEIGHT "${output_dir}/model_${t}.pth"
    done
else
    for t in ${checkpoint_list[@]}
    do
        python tools/relation_test_net.py --config-file "${output_dir}/config.yml" \
                TEST.IMS_PER_BATCH 1 \
                OUTPUT_DIR ${output_dir} \
                MODEL.WEIGHT "${output_dir}/model_${t}.pth"
    done
fi