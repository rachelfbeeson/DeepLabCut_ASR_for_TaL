import numpy
import os

import numpy as np
from sklearn.preprocessing import StandardScaler


class Utterance:
    """
    Holds all the information about each utterance in the TaL corpus including speaker, gender, split etc.
    """
    def __init__(self, id, modality, text, base_path):
        self.id = id
        self.modality = modality
        self.split = ''
        self.text = text
        [self.speaker, self.utt_id] = id.split('-')
        self.gender = self.speaker[2]
        self.duration = None
        self.base_path = base_path
        self.lip_features = []
        self.us_features = []
        self.combined_feats = None
        self.discarded = False

    def feature_combiner(self):
        """ Combines the lip and US features into one matrix, and normalizes it wrt mean and std """
        if len(self.lip_features) and len(self.us_features):
            matrix = numpy.concatenate([self.lip_features, self.us_features], axis=1)
            matrix = numpy.vstack(matrix).astype(float)
            std_slc = StandardScaler()
            X_std = std_slc.fit_transform(matrix)
            if self.id == '73fe-166_aud':
                with open('debug.txt', 'w') as f:
                    numpy.set_printoptions(threshold=numpy.prod(matrix.shape))
                    f.write(numpy.array_str(matrix))
            self.combined_feats = X_std
        else:
            self.discarded = True # to track what was thrown out