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
        self.shared_text = []

    def make_utts(self):
        """
        Walks through the TaL corpus and creates the utterance objects, and builds the list of utterance
        objects.
        If the tal_path points to a file, it will assume the format of utt_id text and try to build
        the utterance objects that way.
        Also builds a list of text shared across modalities to use for making sets.
        """
        silent = []
        modal = []

        if os.path.isdir(self.tal_path):
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
                                utterance = Utterance(utt_name, 'silent', text, base_path)
                                self.utterance_list.append(utterance)
                                silent.append(text)
                            elif 'aud' in utt_name:
                                utterance = Utterance(utt_name, 'modal', text, base_path)
                                self.utterance_list.append(utterance)
                                modal.append(text)
                    else:
                        continue

        elif os.path.isfile(self.tal_path):
            with open(self.tal_path, 'r') as f:
                utterances = f.readlines()
            for line in utterances:
                [utt_id, text] = line.split(' ', 1)
                if 'sil' in utt_id:
                    utterance = Utterance(utt_id, 'silent', text, base_path=None)
                    self.utterance_list.append(utterance)
                    silent.append(text)
                elif 'aud' in utt_id:
                    utterance = Utterance(utt_id, 'modal', text, base_path=None)
                    self.utterance_list.append(utterance)
                    modal.append(text)

        silent = set(silent)
        modal = set(modal)
        self.shared_text = silent.intersection(modal)

    def make_sets(self):
        """
        Determines the split of the utterances. Utterances which share text in different modalities
        are labelled with a test split.
        """
        none_utts = []

        for utt in self.utterance_list:
            if utt.text in self.shared_text:
                if utt.modality == 'silent':
                    utt.split = 'sil_test'
                if utt.modality == 'modal':
                    utt.split = 'mod_test'
            else:
                if utt.modality == 'modal':
                    utt.split = 'train'
                else:
                    none_utts.append(utt)

        for utt in none_utts:
            self.utterance_list.remove(utt)

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
        self.utterance_list.sort(key=lambda x: x.split)
        for split, utts in itertools.groupby(self.utterance_list, key=lambda utt: utt.split):
            if split != '':
                kaldi_file_maker.make_kaldi_files(utts, split)
        kaldi_file_maker.make_language_files()

    def forward(self):
        """ Main driver for the controller, going through all the utterance processing steps. """
        print('===Making utterances from TaL Corpus===')
        self.make_utts()
        self.make_sets()
        if self.make_features:
            print('===Making videos for DLC usage===')
            self.make_videos()
        print('===Extracting and processing features===')
        self.set_features()
        print('===Making files to be used by Kaldi===')
        self.make_kaldi_files()
        print('===FINISHED===')

