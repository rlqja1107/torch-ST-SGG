export CUDA_VISIBLE_DEVICES="0" # if you use multi-gpu, then insert the number of gpu e.g., export CUDA_VISIBLE_DEVICES="4,5"
mutli_gpu=false # whether use multi-gpu or not
num_gpu=2
OUTPATH="checkpoints/50/bgnn/predcls/ie_trans/pretrain" # save directory
GLOVE_DIR="/home/public/Datasets/CV/vg_bm" # Glove directory
mkdir -p $OUTPATH

if $mutli_gpu;then
  python -m torch.distributed.launch \
    --master_port 10093 --nproc_per_node=$num_gpu \
    tools/relation_train_net.py --config-file "configs/sup-50.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR BGNNPredictor_GSL \
    SOLVER.IMS_PER_BATCH 6 \
    TEST.IMS_PER_BATCH 1 \
    SOLVER.MAX_ITER 50000 \
    SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    SOLVER.PRE_VAL False \
    TEST.METRIC "R" \
    MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON True \
    MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS False \
    MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE False \
    OUTPUT_DIR $OUTPATH  \
    GLOVE_DIR $GLOVE_DIR \
    MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING False
else
  python tools/relation_train_net.py --config-file "configs/sup-50_motif_vctree.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR BGNNPredictor_GSL \
    SOLVER.IMS_PER_BATCH 6 \
    TEST.IMS_PER_BATCH 1 \
    SOLVER.MAX_ITER 50000 \
    SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    TEST.METRIC "R" \
    MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON True \
    MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS False \
    MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE False \
    SOLVER.PRE_VAL False \
    OUTPUT_DIR $OUTPATH  \
    GLOVE_DIR $GLOVE_DIR \
    MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING False
fi