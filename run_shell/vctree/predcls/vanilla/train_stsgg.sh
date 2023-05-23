export CUDA_VISIBLE_DEVICES="0" # if you use multi-gpu, then insert the number of gpu e.g., export CUDA_VISIBLE_DEVICES="4,5"
mutli_gpu=false # whether use multi-gpu or not
num_gpu=2
OUTPATH="checkpoints/50/vctree/predcls/base_stsgg" # save directory
GLOVE_DIR="/home/public/Datasets/CV/vg_bm" # Glove directory

#ST-SGG Hyper-parameter
ALPHA_DEC=0.4
ALPHA_INC=0.8
PRETRAINED_WEIGHT_PATH="checkpoints/50/vctree/predcls/base/model_???.pth" # Path of pretrained model
mkdir -p $OUTPATH

if $mutli_gpu;then
  python -m torch.distributed.launch \
    --master_port 10093 --nproc_per_node=$num_gpu \
    tools/relation_train_net.py --config-file "configs/sup-50.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.WEIGHT $PRETRAINED_WEIGHT_PATH \
    MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor_STSGG \
    SOLVER.IMS_PER_BATCH 12 \
    TEST.IMS_PER_BATCH 2 \
    SOLVER.MAX_ITER 50000 \
    SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    OUTPUT_DIR $OUTPATH  \
    MODEL.ROI_RELATION_HEAD.NUM_CLASSES 51 \
    SOLVER.PRE_VAL False \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    TEST.INFERENCE "SOFTMAX" \
    GLOVE_DIR $GLOVE_DIR \
    MODEL.ROI_RELATION_HEAD.STSGG_MODULE.BETA 1.0 \
    MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING False \
    MODEL.ROI_RELATION_HEAD.STSGG_MODULE.ALPHA_DEC $ALPHA_DEC \
    MODEL.ROI_RELATION_HEAD.STSGG_MODULE.ALPHA_INC $ALPHA_INC
else
  python tools/relation_train_net.py --config-file "configs/sup-50.yaml" \
    DATASETS.TRAIN \(\"50DS_VG_VGKB_train\",\) \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.WEIGHT $PRETRAINED_WEIGHT_PATH \
    MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor_STSGG \
    SOLVER.IMS_PER_BATCH 6 \
    TEST.IMS_PER_BATCH 1 \
    SOLVER.MAX_ITER 50000 \
    SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    OUTPUT_DIR $OUTPATH  \
    MODEL.ROI_RELATION_HEAD.NUM_CLASSES 51 \
    SOLVER.PRE_VAL False \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    TEST.INFERENCE "SOFTMAX" \
    GLOVE_DIR $GLOVE_DIR \
    MODEL.ROI_RELATION_HEAD.STSGG_MODULE.BETA 1.0 \
    MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING False \
    MODEL.ROI_RELATION_HEAD.STSGG_MODULE.ALPHA_DEC $ALPHA_DEC \
    MODEL.ROI_RELATION_HEAD.STSGG_MODULE.ALPHA_INC $ALPHA_INC
fi