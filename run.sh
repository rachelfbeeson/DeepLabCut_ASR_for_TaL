#!/bin/bash
. ./path.sh || exit 1
. ./cmd.sh || exit 1
nj=1       # number of parallel jobs
lm_order=2 # language model order (n-gram quantity)
stage=0
rerun_lm=0 # set this to 1 if you want to re-estimate language model from tri3b estimations
# Safety mechanism (possible running this script with modified arguments)
. utils/parse_options.sh || exit 1
[[ $# -ge 1 ]] && { echo "Wrong arguments!"; exit 1; }
if [ $stage -le 0 ]; then
# Removing previously created data (from last run.sh execution)
rm -rf data/train_sp*
rm -rf data/train/split*
rm -rf exp data/train/spk2utt data/train/cmvn.scp data/local/lang data/lang data/lang_old data/local/dict_old data/local/tmp data/local/dict/lexiconp.txt mfcc
fi
if [ $stage -le 1 ]; then
echo
echo "===== PREPARING ACOUSTIC DATA ====="
echo
# Needs to be prepared by hand (or using self written scripts):
#
# spk2gender  [<speaker-id> <gender>]
# wav.scp     [<uterranceID> <full_path_to_audio_file>]
# text        [<uterranceID> <text_transcription>]
# utt2spk     [<uterranceID> <speakerID>]
# corpus.txt  [<text_transcription>]
# Making spk2utt files
utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt

    
echo
echo "===== FEATURES FORMATTING ====="
echo
# Making feats.scp files
mfccdir=mfcc

#copy-feats ark:data/sil_test/feats.txt ark,scp:data/sil_test/feats.ark,data/sil_test/feats.scp
#copy-feats ark:data/mod_test/feats.txt ark,scp:data/mod_test/feats.ark,data/mod_test/feats.scp
copy-feats ark:data/train/feats.txt ark,scp:data/train/feats.ark,data/train/feats.scp

cp data/train/feats.scp mfcc/raw_mfcc_train.1.scp
cp data/train/feats.ark mfcc/raw_mfcc_train.1.ark

#calculate utt2dur

frame_shift=$(cat data/train/frame_shift)
  feat-to-len scp:data/train/feats.scp ark,t:- |
    awk -v frame_shift=$frame_shift '{print $1, ($2)*frame_shift}' >data/train/utt2dur

utils/validate_data_dir.sh data/train --no-wav     # script for checking prepared data - here: for data/train directory
utils/fix_data_dir.sh data/train          # tool for data proper sorting if needed - here: for data/train directory


# Making cmvn.scp files
steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train $mfccdir
echo
echo "===== PREPARING LANGUAGE DATA ====="
echo
# Needs to be prepared by hand (or using self written scripts):
#
# lexicon.txt           [<word> <phone 1> <phone 2> ...]
# nonsilence_phones.txt [<phone>]
# silence_phones.txt    [<phone>]
# optional_silence.txt  [<phone>]
# Preparing language data
utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang
echo
echo "===== LANGUAGE MODEL CREATION ====="
echo "===== MAKING lm.arpa ====="
echo
loc=`which ngram-count`;
if [ -z $loc ]; then
        if uname -a | grep 64 >/dev/null; then
                sdir=$KALDI_ROOT/tools/srilm/bin/i686-m64
        else
                        sdir=$KALDI_ROOT/tools/srilm/bin/i686
        fi
        if [ -f $sdir/ngram-count ]; then
                        echo "Using SRILM language modelling tool from $sdir"
                        export PATH=$PATH:$sdir
        else
                        echo "SRILM toolkit is probably not installed.
                                Instructions: tools/install_srilm.sh"
                        exit 1
        fi
fi
local=data/local
mkdir $local/tmp
ngram-count -order $lm_order -write-vocab $local/tmp/vocab-full.txt -wbdiscount -text $local/corpus.txt -lm $local/tmp/lm.arpa
echo
echo "===== MAKING G.fst ====="
echo
lang=data/lang
arpa2fst --disambig-symbol=#0 --read-symbol-table=$lang/words.txt $local/tmp/lm.arpa $lang/G.fst
fi
if [ $stage -le 2 ]; then
echo
echo "===== MONO TRAINING ====="
echo
steps/train_mono.sh --nj $nj --cmd "$train_cmd" data/train data/lang exp/mono
echo
echo "===== MONO ALIGNMENT ====="
echo
steps/align_si.sh --nj $nj --cmd "$train_cmd" data/train data/lang exp/mono exp/mono_ali || exit 1
fi
if [ $stage -le 3 ]; then
echo
echo "===== TRI1 (first triphone pass) TRAINING ====="
echo
steps/train_deltas.sh --cmd "$train_cmd" 2000 11000 data/train data/lang exp/mono_ali exp/tri1 || exit 1
echo
echo "===== TRI1 (first triphone pass) ALIGN ====="
steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri1 exp/tri1_ali
fi
if [ $stage -le 4 ]; then
echo "===== TRI2 (second triphone pass) TRAIN ====="
steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
    data/train data/lang exp/tri1_ali exp/tri2b
echo "===== TRI2 (second triphone pass) ALIGN ====="

steps/align_si.sh  --nj $nj --cmd "$train_cmd" --use-graphs true \
    data/train data/lang exp/tri2b exp/tri2b_ali
fi
if [ $stage -le 5 ]; then
echo "===== TRI3 (SAT triphone pass) TRAIN ====="
steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
    data/train data/lang exp/tri2b_ali exp/tri3b
     
if [ $rerun_lm -eq 1 ]; then
echo "===== REBUILDING LM ====="
# save the old lang model info to another directory
cp data/local/dict data/local/dict_old
cp data/lang data/lang_old

steps/get_prons.sh --cmd "$train_cmd" \
    data/train data/lang_old exp/tri3b
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_old \
    exp/tri3b/pron_counts_nowb.txt exp/tri3b/sil_counts_nowb.txt \
    exp/tri3b/pron_bigram_counts_nowb.txt data/local/dict 

  utils/prepare_lang.sh data/local/dict \
    "<UNK>" data/local/lang data/lang


if [ -z $loc ]; then
        if uname -a | grep 64 >/dev/null; then
                sdir=$KALDI_ROOT/tools/srilm/bin/i686-m64
        else
                        sdir=$KALDI_ROOT/tools/srilm/bin/i686
        fi
        if [ -f $sdir/ngram-count ]; then
                        echo "Using SRILM language modelling tool from $sdir"
                        export PATH=$PATH:$sdir
        else
                        echo "SRILM toolkit is probably not installed.
                                Instructions: tools/install_srilm.sh"
                        exit 1
        fi
fi

local=data/local
arpa2fst --disambig-symbol=#0 --read-symbol-table=$lang/words.txt $local/tmp/lm.arpa $lang/G.fst
fi
echo "===== TRI3 (SAT triphone pass) ALIGN ====="
steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/train $lang exp/tri3b exp/tri3b_ali
    

fi
if [ $stage -le 6 ]; then
echo "===== TDNN TRAIN ====="
local/nnet2/run_5c.sh --feat_type raw
echo "===== run.sh script is finished ====="
echo
fi























