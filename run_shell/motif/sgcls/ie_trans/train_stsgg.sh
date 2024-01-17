export CUDA_VISIBLE_DEVICES="0" # if you use multi-gpu, then insert the number of gpu e.g., export CUDA_VISIBLE_DEVICES="4,5"
mutli_gpu=false # whether use multi-gpu or not
num_gpu=2
OUTPATH="checkpoints/50/motif/sgcls/st_sgg" # save directory
GLOVE_DIR="Glove" # Glove directory
SPECIFIED_PATH="datasets/50/motif/Motif_i-trans.pk"
ALPHA_DEC=0.4
ALPHA_INC=0.4
MODEL_WEIGHT_PATH="checkpoints/50/motif/sgcls/no_ext_yes_int/model_???.pth" # Path of pretrained model
mkdir -p $OUTPATH

if $mutli_gpu;then
  python -m torch.distributed.launch \
    --master_port 10093 --nproc_per_node=$num_gpu \
    tools/relation_train_net.py --config-file "configs/wsup-50.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.WEIGHT $MODEL_WEIGHT_PATH \
    MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor_STSGG \
    SOLVER.IMS_PER_BATCH 12 \
    TEST.IMS_PER_BATCH 2 \
    SOLVER.MAX_ITER 50000 \
    SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    OUTPUT_DIR $OUTPATH  \
    MODEL.ROI_RELATION_HEAD.NUM_CLASSES 51 \
    WSUPERVISE.SPECIFIED_DATA_FILE $SPECIFIED_PATH \
    EM.MODE "x" \
    GLOVE_DIR $GLOVE_DIR \
    MODEL.ROI_RELATION_HEAD.STSGG_MODULE.BETA 1.0 \
    MODEL.ROI_RELATION_HEAD.STSGG_MODULE.ALPHA_DEC $ALPHA_DEC \
    MODEL.ROI_RELATION_HEAD.STSGG_MODULE.ALPHA_INC $ALPHA_INC
else
  python tools/relation_train_net.py --config-file "configs/wsup-50.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.WEIGHT $PRETRAINED_WEIGHT_PATH \
    MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor_STSGG \
    SOLVER.IMS_PER_BATCH 6 \
    TEST.IMS_PER_BATCH 1 \
    SOLVER.MAX_ITER 50000 \
    SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    OUTPUT_DIR $OUTPATH  \
    MODEL.ROI_RELATION_HEAD.NUM_CLASSES 51 \
    WSUPERVISE.SPECIFIED_DATA_FILE $SPECIFIED_PATH \
    EM.MODE "x" \
    GLOVE_DIR $GLOVE_DIR \
    MODEL.ROI_RELATION_HEAD.STSGG_MODULE.BETA 1.0 \
    MODEL.ROI_RELATION_HEAD.STSGG_MODULE.ALPHA_DEC $ALPHA_DEC \
    MODEL.ROI_RELATION_HEAD.STSGG_MODULE.ALPHA_INC $ALPHA_INC
fi