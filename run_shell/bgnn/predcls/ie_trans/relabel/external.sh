export CUDA_VISIBLE_DEVICES="0"
OUTPATH="checkpoints/50/bgnn/predcls/ie_trans"
MODEL_WEIGHT_PATH="checkpoints/50/bgnn/predcls/ie_trans/pretrain/model_???.pth" # Insert the pretraiend model weight on predcls task
GLOVE_DIR="Glove" # Glove directory

mkdir -p $OUTPATH


python tools/external_relabel.py --config-file "configs/wsup-50_external.yaml" \
  MODEL.WEIGHT $MODEL_WEIGHT_PATH \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR BGNNPredictor_GSL \
  SOLVER.IMS_PER_BATCH 6 TEST.IMS_PER_BATCH 1 \
  SOLVER.MAX_ITER 50000 \
  SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  OUTPUT_DIR $OUTPATH  \
  MODEL.ROI_RELATION_HEAD.NUM_CLASSES 51 \
  SOLVER.PRE_VAL False \
  MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON True \
  GLOVE_DIR $GLOVE_DIR \
  MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS False \
  MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE False \
  MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.PRETRAIN_RELNESS_MODULE False \
  WSUPERVISE.DATASET ExTransDataset  EM.MODE E  WSUPERVISE.SPECIFIED_DATA_FILE  datasets_vg/vg/50/vg_clip_logits.pk


cp tools/ietrans/external_cut.py $OUTPATH
cp tools/ietrans/na_score_rank.py $OUTPATH
cd $OUTPATH
python na_score_rank.py
python external_cut.py 1.0 # use all relabeled DS data
