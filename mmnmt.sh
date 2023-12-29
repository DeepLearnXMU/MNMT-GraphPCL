#!/bin/bash
#set -xe
export OMP_NUM_THREADS=1

gpu=0

tgt=de
maxsteps=80000

if [ $tgt == 'fr' ];then
maxsteps=100000
fi


tau=0.1
hardtau=0.1
cllamb=0.01
hardn=10
start=40000
hardtype=ali
seed=1234

modelname=


CUDA_VISIBLE_DEVICES=$gpu python3 -u $RUN/main.py --model ${modelname} \
    --corpus_prex $datapath/train.bpe --lang en $tgt graph \
    --main_path $RUN \
    --sfgtype ind --pace 1.0 \
    --hardtau $hardtau --hardins_start $start --maxnum_neggraph $hardn \
    --clproj ffn2 --mtrep $mtrep --tau $tau --cllamb $cllamb \
    --wocl $wocl --hardtype $hardtype \
    --valid $datapath/val.bpe --img_dp 0.5 --objdim 2048 --seed $seed \
    --boxprobs $datapath/boxporbs.pkl --n_enclayers 3 \
    --writetrans $RUN/decoding/${modelname}.devtrans --boxfeat $datapath/train.res.pkl $datapath/val.res.pkl \
    --ref $datapath/val.lc.norm.tok.$tgt --batch_size 4000 --delay 1 --warmup 4000 \
    --vocab $datapath --vocab_size 40000 --load_vocab --smoothing 0.1 --share_embed --share_vocab --beam_size 4 \
    --params user --lr 1.0 --init standard --enc_dp 0.5 --dec_dp 0.5 --input_drop_ratio 0.5 \
    --n_layers 4 --n_heads 4 --d_model 128 --d_hidden 256 \
    --max_len 100 --eval_every 2000 --save_every 5000 --maximum_steps $maxsteps >$RUN/${modelname}.train 2>&1

#bash $RUN/test.sh $modelname $tgt $gpu

