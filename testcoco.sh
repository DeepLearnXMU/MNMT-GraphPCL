#!/bin/bash
datapath=data/bpe_data

modelname=$1
gpu=$2
tgt=de

CUDA_VISIBLE_DEVICES=$gpu python main.py --mode test --load_from models/${modelname} \
--boxfeat $datapath/test_2017_mscoco.res.pkl \
--boxprobs $datapath/cocoboxprobs.pkl --test $datapath/test_2017_mscoco.bpe --ref $datapath/test_2017_mscoco.lc.norm.tok.$tgt \
--writetrans decoding/${modelname}.2017coco.b4trans --beam_size 4 | tee -a ${modelname}.tranlog

# cd multeval-0.5.1
# bash run.sh $datapath/test_2017_mscoco.lc.norm.tok.$tgt ../decoding/$modelname.2017coco.b4trans $tgt >>../$modelname.tranlog



