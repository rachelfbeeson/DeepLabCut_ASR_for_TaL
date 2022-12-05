#!/usr/bin/env bash

#pip install deeplabcut[tf]

#git clone https://github.com/articulateinstruments/DeepLabCut-for-Speech-Production.git

other_files=DeepLabCut-for-Speech-Production/Installation_Instructions/Other_Files



US_project=$PWD/DeepLabCut-for-Speech-Production/Ultrasound
US_test_resnet=$US_project/dlc-models/iteration-0/SpeechProductionFeb12-trainset34shuffle0/test/
mkdir $US_test_resnet
US_train_resnet=$US_project/dlc-models/iteration-0/SpeechProductionFeb12-trainset34shuffle0/train/
US_test_mobile=$US_project/dlc-models/iteration-0/SpeechProductionFeb12-trainset34shuffle1/test/
mkdir $US_test_mobile
US_train_mobile=$US_project/dlc-models/iteration-0/SpeechProductionFeb12-trainset34shuffle1/train/
Lips_project=$PWD/DeepLabCut-for-Speech-Production/Lips
Lips_test_resnet=$Lips_project/dlc-models/iteration-0/Tal_LipsJan28-trainset35shuffle0/test/
mkdir $Lips_test_resnet
Lips_train_resnet=$Lips_project/dlc-models/iteration-0/Tal_LipsJan28-trainset35shuffle0/train/
Lips_test_mobile=$Lips_project/dlc-models/iteration-0/Tal_LipsJan28-trainset35shuffle1/test/
mkdir $Lips_test_mobile
Lips_train_mobile=$Lips_project/dlc-models/iteration-0/Tal_LipsJan28-trainset35shuffle1/train/


for file in $other_files/auto*; do 
sed -i 's.\\./.g' $file
done

sed 's.REPLACE_PATH.'$US_project'.g' $other_files/autoConfigUltrasound > $US_project/config.yaml
sed 's.REPLACE_PATH.'$US_project'.g' $other_files/autoPoseCfgUSResNetTest > $US_test_resnet/pose_cfg.yaml
sed 's.REPLACE_PATH.'$US_project'.g' $other_files/autoPoseCfgUSResNetTrain > $US_train_resnet/pose_cfg.yaml
sed 's.REPLACE_PATH.'$US_project'.g' $other_files/autoPoseCfgUSMobileNetTest > $US_test_mobile/pose_cfg.yaml
sed 's.REPLACE_PATH.'$US_project'.g' $other_files/autoPoseCfgUSMobileNetTrain > $US_train_mobile/pose_cfg.yaml
sed 's.REPLACE_PATH.'$Lips_project'.g' $other_files/autoConfigLips > $Lips_project/config.yaml
sed 's.REPLACE_PATH.'$Lips_project'.g' $other_files/autoPoseCfgLipsResNetTest > $Lips_test_resnet/pose_cfg.yaml
sed 's.REPLACE_PATH.'$Lips_project'.g' $other_files/autoPoseCfgLipsResNetTrain > $Lips_train_resnet/pose_cfg.yaml
sed 's.REPLACE_PATH.'$Lips_project'.g' $other_files/autoPoseCfgLipsMobileNetTest > $Lips_test_mobile/pose_cfg.yaml
sed 's.REPLACE_PATH.'$Lips_project'.g' $other_files/autoPoseCfgLipsMobilenetTrain > $Lips_train_mobile/pose_cfg.yaml

#cp $other_files/autoConfigUltrasound $US_project/config.yaml
#cp $other_files/autoPoseCfgUSResNetTest $US_test_resnet/pose_cfg.yaml
#cp $other_files/autoPoseCfgUSResNetTrain $US_train_resnet/pose_cfg.yaml
#cp $other_files/autoPoseCfgUSMobileTest $US_test_mobile/pose_cfg.yaml
#cp $other_files/autoPoseCfgUSMobileTrain $US_train_mobile/pose_cfg.yaml
#cp $other_files/autoConfigLips $Lips_project/config.yaml
#cp $other_files/autoPoseCfgLipsResNetTest $Lips_test_resnet/pose_cfg.yaml
#cp $other_files/autoPoseCfgLipsResNetTrain $Lips_train_resnet/pose_cfg.yaml
#cp $other_files/autoPoseCfgLipsMobileTest $Lips_test_mobile/pose_cfg.yaml
#cp $other_files/autoPoseCfgLipsMobileTrain $Lips_train_mobile/pose_cfg.yaml
