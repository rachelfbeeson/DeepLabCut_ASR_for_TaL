import os
import itertools
from tools.KaldiFileMaker import KaldiFileMaker
from tools.VideoMaker import VideoMaker
from tools.FeatureMaker import FeatureMaker
from tools.config_manager import config
from tools.Utterance import Utterance


class UtteranceController:
    """
    Creates utterance objects that are found in the TaL Path and maintains a list of them.
    This list is used for further operations on the utterances, i.e. preparing videos for DLC,
    running DLC, creating features, and creating Kaldi files.
    """
    def __init__(self, make_features=True):
        self.tal_path = config.get('Paths','tal_path')
        self.video_path = config.get('Paths', 'video_and_csv_path')
        self.us_video_path = os.path.join(self.video_path, 'USVideo')
        self.lip_video_path = os.path.join(self.video_path, 'LipVideo')
        self.dlc_project = config.get('Paths', 'DLC_project')
        self.utterance_list = []
        self.tongue_anatomy = ['vallecula', 'tongueRoot1', 'tongueRoot2', 'tongueBody1', 'tongueBody2', 'tongueDorsum1',
                  'tongueDorsum2', 'tongueBlade1', 'tongueBlade2', 'tongueTip1', 'tongueTip2', 'hyoid',
                  'mandible', 'shortTendon']
        self.lip_anatomy = ['leftLip', 'rightLip', 'topleftinner', 'bottomleftinner', 'toprightinner', 'bottomrightinner',
               'topmidinner', 'bottommidinner']
        self.make_features = make_features

    def utterance_sets(self):
        """
        Walks through the TaL corpus and creates the utterance objects, determining which split they belong to.
        For the purposes of the original experiment, the two test splits are all
        the utterances which share the same text in both silent and
        audible modalities.
        """
        sil_dict = {}
        mod_dict = {}

        for folder in os.listdir(self.tal_path):
            folder_name = os.path.join(self.tal_path, folder)
            for file_name in os.listdir(folder_name):
                [utt_id, ext] = file_name.split('.')
                # should be one param file for every utterance
                if ext == 'param':
                    base_path = os.path.join(folder_name, utt_id)
                    utt_name = folder + '-' + utt_id
                    with open(base_path + '.txt', 'r') as f:
                        text = f.readline()
                        if 'sil' in utt_name:
                            sil_dict[utt_name] = text
                        elif 'aud' in utt_name:
                            mod_dict[utt_name] = text
                else:
                    continue

        silent = set(sil_dict.values())
        modal = set(mod_dict.values())
        shared = silent.intersection(modal)

        for utt in mod_dict:
            if mod_dict[utt] in shared:
                utterance = Utterance(utt,'mod_test',mod_dict[utt],self.tal_path)
                self.utterance_list.append(utterance)
            else:
                utterance = Utterance(utt,'train', mod_dict[utt], self.tal_path)
                self.utterance_list.append(utterance)
        for utt in sil_dict:
            if sil_dict[utt] in shared:
                utterance = Utterance(utt, 'sil_test', sil_dict[utt], self.tal_path)
                self.utterance_list.append(utterance)

    def make_videos(self):
        """ Make videos of all the utterances in the list """
        video_maker = VideoMaker(self.us_video_path, self.lip_video_path)
        video_maker.video_handler(self.utterance_list)

    def set_features(self):
        """ Creates Lip and Ultrasound features for each utterance, which are combined to create one feature matrix. """
        US_feature_maker = FeatureMaker(self.us_video_path, self.tongue_anatomy,
                                        os.path.join(self.dlc_project, 'Ultrasound'))
        lip_feature_maker = FeatureMaker(self.lip_video_path, self.lip_anatomy,
                                         os.path.join(self.dlc_project, 'Lips'))
        if self.make_features:
            lip_feature_maker.run_DLC()
            US_feature_maker.run_DLC()
        for utterance in self.utterance_list:
            US_feature_maker.process_features(utterance)
            utterance.us_features = US_feature_maker.features
            lip_feature_maker.process_features(utterance)
            utterance.lip_features = lip_feature_maker.features
            utterance.feature_combiner()

    def make_kaldi_files(self):
        """ Creates the necessary Kaldi files from the splits determined earlier. """
        kaldi_file_maker = KaldiFileMaker()
        kaldi_file_maker.make_dirs()
        for split, utts in itertools.groupby(self.utterance_list, key=lambda utt: utt.split):
            kaldi_file_maker.make_kaldi_files(utts, split)
        kaldi_file_maker.make_language_files()

    def forward(self):
        """ Main driver for the controller, going through all the utterance processing steps. """
        print('===Making utterances from TaL Corpus===')
        self.utterance_sets()
        if self.make_features:
            print('===Making videos for DLC usage===')
            self.make_videos()
        print('===Extracting and processing features===')
        self.set_features()
        print('===Making files to be used by Kaldi===')
        self.make_kaldi_files()
        print('===FINISHED===')

