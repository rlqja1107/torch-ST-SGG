OUTPATH="checkpoints/50/motif/predcls/ie_trans"
mkdir -p $OUTPATH
cp tools/ietrans/combine.py $OUTPATH
cd $OUTPATH
python combine.py
