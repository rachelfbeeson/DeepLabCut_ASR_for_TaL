#!/bin/bash
. ./path.sh || exit 1
. ./cmd.sh || exit 1

fmllr=0
dir=exp/nnet5c

for test in sil_test mod_test; do

cp data/$test/feats.scp mfcc/raw_mfcc_test.1.scp
cp data/$test/feats.ark mfcc/raw_mfcc_test.1.ark

utils/utt2spk_to_spk2utt.pl data/$test/utt2spk > data/$test/spk2utt

utils/validate_data_dir.sh data/$test --no-wav    # script for checking prepared data

utils/fix_data_dir.sh data/$test 

# Making cmvn.scp files

steps/compute_cmvn_stats.sh data/$test exp/make_mfcc/$test $mfccdir

utils/mkgraph.sh data/lang exp/tri3b exp/tri3b/graph || exit 1

if [ $fmllr -eq 0 ]; then

# vanilla feature decoding

steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 4 \
   exp/tri3b/graph data/$test $dir/decode_$test
   
fi
   
if [ $fmllr -eq 1 ]; then

# getting fmllr features

steps/decode_raw_fmllr.sh --nj 4 --cmd "$decode_cmd" \
        exp/tri3b/graph data/$test \
        exp/tri3b/decode_fmllr_$test || exit 1;
        
# decoding the fmllr features from the above using the model

steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 4 \
   --transform-dir exp/tri3b/decode_fmllr_$test \
    exp/tri3b/graph data/$test $dir/decode_fmllr_$test
    
fi
done
