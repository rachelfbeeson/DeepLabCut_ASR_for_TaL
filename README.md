# About

The purpose of this repo is to run the experiment detailed in *Multi-speaker Speech Recognition using Articulator Pose Estimation on Silent versus Voiced Speech*. In this experiment, ASR is performed not from the sound produced while speaking, but from the articulators. Video of the lips and ultrasound of the tongue are taken from the TaL corpus. These videos are run through a pre-trained pose-estimation model in DeepLabCut, in order to extract the location of the x and y coordinates of the articulators over time. These series of coordinates over time are transformed into feature vectors which are fed into a DNN-HMM system in Kaldi.

In the original experiment, two test sets were split from the TaL corpus, one made of utterances spoken silently and another made of utterances spoken aloud. The two test sets were designed to have the same text represented in each modality. This repo produces the same training and test split.

# Instructions

## 1. Prerequisites

Before you can use this repo, there are some prerequisites.

### a. TaL Corpus

If you plan to extract your own features, you must first download the TaL corpus. The corpus can be accessed here: https://ultrasuite.github.io/data/tal_corpus/#download

### b. DeepLabCut for Speech

In order to extract the features, you must also install DeepLabCut and download the requisite models from this repository: https://github.com/articulateinstruments/DeepLabCut-for-Speech-Production/tree/main/Installation_Instructions

The repository provides an easy installation for systems running Windows. If you would like to download and configure the model on Ubuntu, I have provided a script (dlc.sh) which should do the installation and configuration for you.

### c. Kaldi

You must install Kaldi to run the experiment. See instructions here: https://kaldi-asr.org/doc/install.html

After you have installed Kaldi, this repo should be placed in the kaldi/egs directory.

### d. Optional: The BEEP dictionary

The experiment originally uses the BEEP dictionary to build the lexicon, as most speakers in the TaL corpus are British. You can download it here: https://www.openslr.org/14/

Alternatively, setting the 'beep' option in conf.ini, you can choose to build the lexicon using the CMUdict in nltk. You do not need to download anything for this.

## 2. Running the experiment

### a. Editing the conf.ini

At a minimum, you must edit the conf.ini to contain the correct 'Paths'. The other options are set to the original experiment options, but you can change these as you wish.

### b. Placing the Utils and Steps directories

This experiment uses the /utils and /steps directiories from the wsj egs in Kaldi. In order to properly run the experiment, you will need to either create a symlink to these directories, or copy them into tal_dlc.

To create a symlink from inside tal_dlc to wsj:

 ln -s ../wsj/s5/steps steps

### c. Running the experiment

To run, from the terminal, do:
python main_setup.py

After it completes, do:
./run.sh

To test the model, do:
./test.sh

