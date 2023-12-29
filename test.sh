#!/bin/bash
datapath=data/bpe_data

RUN=

datapath=bpe_data

modelname=${1}
tgt=${2}
gpu=${3}

CUDA_VISIBLE_DEVICES=$gpu python3 $RUN/main.py --mode test --load_from $RUN/models/${modelname} \
--boxfeat $datapath/test_2017_mscoco.res.pkl \
--boxprobs $datapath/cocoboxprobs.pkl --test $datapath/test_2017_mscoco.bpe \
--ref $datapath/test_2017_mscoco.lc.norm.tok.$tgt \
--writetrans $RUN/decoding/${modelname}.2017coco.b4trans --beam_size 4 >>$RUN/${modelname}.tranlog 2>&1


CUDA_VISIBLE_DEVICES=$gpu python -u $RUN/main.py --mode test --load_from $RUN/models/${modelname} \
--test $datapath/test_2016_flickr.bpe --ref $datapath/test_2016_flickr.lc.norm.tok.$tgt \
--boxfeat $datapath/test_2016_flickr.res.pkl --boxprobs $datapath/boxporbs.pkl \
--writetrans $RUN/decoding/${modelname}.2016.$tgt.b4trans --beam_size 4 >>$RUN/${modelname}.tranlog 2>&1

CUDA_VISIBLE_DEVICES=$gpu python -u $RUN/main.py --mode test --load_from $RUN/models/${modelname} \
--test $datapath/test_2017_flickr.bpe --ref $datapath/test_2017_flickr.lc.norm.tok.$tgt \
--boxfeat $datapath/test_2017_flickr.res.pkl --boxprobs $datapath/boxporbs.pkl \
--writetrans $RUN/decoding/${modelname}.2017.$tgt.b4trans --beam_size 4 >>$RUN/${modelname}.tranlog 2>&1

if [ $tgt == 'de' ]; then
CUDA_VISIBLE_DEVICES=$gpu python3 $RUN/main.py --mode test --load_from $RUN/models/${modelname} \
--boxfeat $datapath/test_2017_mscoco.res.pkl \
--boxprobs $datapath/cocoboxprobs.pkl --test $datapath/test_2017_mscoco.bpe \
--ref $datapath/test_2017_mscoco.lc.norm.tok.$tgt \
--writetrans $RUN/decoding/${modelname}.2017coco.b4trans --beam_size 4 >>$RUN/${modelname}.tranlog 2>&1
fi

#cd multeval-0.5.1
#bash run.sh $datapath/test_2016_flickr.lc.norm.tok.$tgt ../decoding/$modelname.2016.$tgt.b4trans $tgt >>../$modelname.tranlog
#bash run.sh $datapath/test_2017_flickr.lc.norm.tok.$tgt ../decoding/$modelname.2017.$tgt.b4trans $tgt >>../$modelname.tranlog

