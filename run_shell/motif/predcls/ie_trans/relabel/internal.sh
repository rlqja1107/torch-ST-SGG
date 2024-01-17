export CUDA_VISIBLE_DEVICES="6"
OUTPATH="checkpoints/50/motif/predcls/ie_trans" # put the pre-trained model's path
MODEL_WEIGHT_PATH="checkpoints/50/motif/predcls/ie_trans/pretrain/model_???.pth" # Insert the pretraiend model weight on predcls task

GLOVE_DIR="Glove" # Glove directory


mkdir -p $OUTPATH

python tools/internal_relabel.py --config-file "configs/wsup-50_internal.yaml" \
  MODEL.WEIGHT $MODEL_WEIGHT_PATH \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
  SOLVER.IMS_PER_BATCH 6 TEST.IMS_PER_BATCH 1 \
  SOLVER.MAX_ITER 50000 \
  SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  MODEL.PRETRAINED_DETECTOR_CKPT maskrcnn_benchmark/pretrained/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR $OUTPATH  \
  MODEL.ROI_RELATION_HEAD.NUM_CLASSES 51 \
  SOLVER.PRE_VAL False \
  GLOVE_DIR $GLOVE_DIR \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS False \
  MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE False \
  MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON False \
  MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.PRETRAIN_RELNESS_MODULE False \
  WSUPERVISE.DATASET InTransDataset  EM.MODE E  WSUPERVISE.SPECIFIED_DATA_FILE  datasets_vg/vg/50/vg_sup_data.pk



cp tools/ietrans/internal_cut.py $OUTPATH
cd $OUTPATH
python internal_cut.py 0.7

