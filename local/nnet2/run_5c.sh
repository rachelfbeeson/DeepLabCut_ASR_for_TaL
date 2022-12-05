#!/usr/bin/env bash

# This is neural net training on top of raw or adapted 40-dimensional features.
#
feat_type=
train_stage=-10
use_gpu=false

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
  fi
  parallel_opts="--gpu 1"
  num_threads=1
  minibatch_size=512
  dir=exp/nnet5c_gpu
else
  num_threads=16
  parallel_opts="--num-threads $num_threads"
  dir=exp/nnet5c
  minibatch_size=128
fi

if [ ! -f $dir/final.mdl ]; then

  steps/nnet2/train_tanh_fast.sh --stage $train_stage \
    --num-threads "$num_threads" \
    --parallel-opts "$parallel_opts" \
    --minibatch-size "$minibatch_size" \
    --num-jobs-nnet 4 \
    --samples-per-iter 400000 \
    --mix-up 8000 \
    --initial-learning-rate 0.01 --final-learning-rate 0.001 \
    --num-hidden-layers 4 --hidden-layer-dim 1024 \
    --cmd "$decode_cmd" \
    --feat_type $feat_type \
     data/train data/lang exp/tri3b_ali $dir || exit 1
fi


