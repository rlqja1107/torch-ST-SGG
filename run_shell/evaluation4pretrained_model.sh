export CUDA_VISIBLE_DEVICES="1" # if using multi-gpu, put the multi gpu numbers, e.g., "2,3"
num_gpu=1 # gpu number
multi_gpu=false


export output_dir="configs/evaluation_config/motif_stsgg_sgcls_eval" # Put the directory for evaluation

if $multi_gpu;then
    python -m torch.distributed.launch --master_port 10093 --nproc_per_node=$num_gpu \
        tools/relation_test_net.py --config-file "${output_dir}/config.yml" \
        TEST.IMS_PER_BATCH 2 \
        OUTPUT_DIR ${output_dir}
else
    python tools/relation_test_net.py --config-file "${output_dir}/config.yml" \
        TEST.IMS_PER_BATCH 1 \
        OUTPUT_DIR ${output_dir}
fi