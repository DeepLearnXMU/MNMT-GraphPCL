datapath=bpe_data

modelname=${1}
tgt=${2}

#bash run.sh $datapath/test_2016_flickr.lc.norm.tok.$tgt ../decoding/$modelname.2016.$tgt.b4trans $tgt >>../$modelname.tranlog
#bash run.sh $datapath/test_2017_flickr.lc.norm.tok.$tgt ../decoding/$modelname.2017.$tgt.b4trans $tgt >>../$modelname.tranlog

multeval/multeval eval --refs $datapath/test_2016_flickr.lc.norm.tok.$tgt --hyps-baseline decoding/$modelname.2016.$tgt.b4trans --meteor.language $tgt >>$modelname.mult
multeval/multeval eval --refs $datapath/test_2017_flickr.lc.norm.tok.$tgt --hyps-baseline decoding/$modelname.2017.$tgt.b4trans --meteor.language $tgt >>$modelname.mult

if [ $tgt == 'de' ]; then
multeval/multeval eval --refs $datapath/test_2017_mscoco.lc.norm.tok.$tgt --hyps-baseline decoding/$modelname.2017coco.b4trans --meteor.language $tgt >>$modelname.mult
fi
