export CUDA_VISIBLE_DEVICES="0" # if you use multi-gpu, then insert the number of gpu e.g., export CUDA_VISIBLE_DEVICES="4,5"
mutli_gpu=false # whether use multi-gpu or not
num_gpu=2

OUTPATH="checkpoints/50/bgnn/sgdet/no_ext_yes_int"
SPECIFIED_PATH="datasets/50/bgnn/BGNN_i-trans.pk"
GLOVE_DIR="Glove" # Glove directory

mkdir -p $OUTPATH
# cp $EXP/50/motif/predcls/lt/combine/relabel/em_E.pk $OUTPATH/em_E.pk


if $mutli_gpu;then
  python -m torch.distributed.launch \
    --master_port 10093 --nproc_per_node=$num_gpu \
    tools/relation_train_net.py --config-file "configs/wsup-50.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR BGNNPredictor_GSL \
    SOLVER.IMS_PER_BATCH 12 \
    TEST.IMS_PER_BATCH 2 \
    SOLVER.MAX_ITER 50000 \
    SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    OUTPUT_DIR $OUTPATH  \
	  MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON True \
    MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS False \
    MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE False \
    MODEL.ROI_RELATION_HEAD.NUM_CLASSES 51 \
    GLOVE_DIR $GLOVE_DIR \
    WSUPERVISE.SPECIFIED_DATA_FILE $SPECIFIED_PATH \
    EM.MODE "x"
else
  python tools/relation_train_net.py --config-file "configs/wsup-50.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR BGNNPredictor_GSL \
    SOLVER.IMS_PER_BATCH 6 \
    TEST.IMS_PER_BATCH 1 \
    SOLVER.MAX_ITER 50000 \
    SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    OUTPUT_DIR $OUTPATH  \
	  MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON True \
    MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS False \
    MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE False \
    MODEL.ROI_RELATION_HEAD.NUM_CLASSES 51 \
    GLOVE_DIR $GLOVE_DIR \
    WSUPERVISE.SPECIFIED_DATA_FILE $SPECIFIED_PATH \
    EM.MODE "x"
fi