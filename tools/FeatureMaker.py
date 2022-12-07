import os
import glob
import deeplabcut
import pandas
import numpy
import scipy.signal as sig
import statistics
from tools.config_manager import config


class FeatureMaker:
    """
    Handles running DLC and processing the features from the CSV files DLC produces.
    The video folder, anatomy, and dlc_project are specific to either the Lips
    or the Ultrasound videos.
    Currently, features can be filtered by DLC likelihood values,
    number of SDs from the mean, and by a low-pass filter.
    Features for one utterance are stored on the object after processing as .features.
    """
    def __init__(self, video_folder, anatomy, dlc_project):
        self.anatomy = anatomy
        self.likelihood_filter = config.getboolean('PostDLC','likelihood')
        self.likelihood_cutoff = config.getfloat('PostDLC','likelihood_cutoff')
        self.outlier_filter = config.getboolean('PostDLC','outlier')
        self.outlier_cutoff = config.getint('PostDLC','outlier_cutoff')
        self.lowpass_filter = config.getboolean('PostDLC','lowpass')
        self.lowpass_cutoff = config.getint('PostDLC','lowpass_cutoff')
        self.dlc_shuffle = config.getint('DLC','shuffle')
        self.dlc_project = dlc_project
        self.video_folder = video_folder
        self.features = pandas.DataFrame()

    def run_DLC(self):
        """ Runs DLC either for Lips or US depending on object instantiation """
        dlc_config = os.path.join(self.dlc_project, 'config.yaml')
        deeplabcut.analyze_videos(dlc_config, self.video_folder, shuffle=self.dlc_shuffle, save_as_csv=True)

    def process_features(self, utterance):
        """ Grabs CSV file output by DLC for a particular utterance and creates features """
        if utterance.base_path is None:
            format_utt = utterance.id.replace('-', '_', 1)
        else:
            format_utt = utterance.id
        if glob.glob(os.path.join(self.video_folder, f'{format_utt}*.csv')):
            csv_file = glob.glob(os.path.join(self.video_folder, f'{format_utt}*.csv'))[0]
            self.features = pandas.read_csv(csv_file,
                                        header=[1, 2])  # headers are two parts, anatomy and then x, y or likelihood
            self.feature_maker()

    def feature_maker(self):
        """ Driver for feature manipulation. """
        if self.likelihood_filter:
            self.likelihood_filtering()
        if self.outlier_filter:
            self.outlier_filtering()
        if self.lowpass_filter:
            self.low_pass_filtering()
        self.features.drop([(part, 'likelihood') for part in self.anatomy], axis=1, inplace=True)
        if self.features.isnull().values.any():
            self.features = pandas.DataFrame()
        self.features = self.features.to_numpy()

    def likelihood_filtering(self):
        """
         Takes the features and replaces values which have an associated likelihood of < .1 with NaN.
         Likelihood is a value output from DLC and represents confidence about a prediction.
         The NaNs are interpolated over.
         """
        for part in self.anatomy:
            self.features[(part, 'x')].where(self.features[(part, 'likelihood')] > self.likelihood_cutoff,
                                             other = numpy.nan, inplace=True)
            self.features[(part, 'y')].where(self.features[(part, 'likelihood')] > self.likelihood_cutoff,
                                             other = numpy.nan, inplace=True)
        self.features.interpolate(axis=0, limit_direction='both', inplace=True)

    def outlier_filtering(self):
        """ Interpolates over values which are > 3 * sd away from the mean."""
        for part in self.anatomy:
            std_x = statistics.stdev(self.features[(part, 'x')])
            mean_x = statistics.mean(self.features[(part, 'x')])
            std_y = statistics.stdev(self.features[(part, 'y')])
            mean_y = statistics.mean(self.features[(part, 'y')])
            self.features[(part, 'x')].where((self.features[(part, 'x')] < mean_x + (std_x * self.outlier_cutoff)) &
                                             (self.features[(part, 'x')] > mean_x - (std_x * self.outlier_cutoff)),
                                             other = numpy.nan, inplace = True)
            self.features[(part, 'y')].where((self.features[(part, 'y')] < mean_y + (std_y * self.outlier_cutoff)) &
                                             (self.features[(part, 'y')] > mean_y - (std_y * self.outlier_cutoff)),
                                             other=numpy.nan, inplace=True)
        self.features.interpolate(axis=0, limit_direction='both', inplace=True)

    def low_pass_filtering(self):
        """
        Low pass filters the features using a 3rd order butterworth filter with a chosen threshold.
        Articulators have been observed to not exceed 9 syll per second.
        (See Knuijt et al. "Reference values of maximum performance tests of speech production")
        Any other "jitter" then could be considered noise introduced from processing.
        """
        sos = sig.butter(3, self.lowpass_cutoff, output='sos', fs=60)
        for part in self.anatomy:
            prefix_x = self.features[(part, 'x')][(self.features.index < 15)]
            prefix_y = self.features[(part, 'y')][(self.features.index < 15)]
            # sosfilt results in initial wave artefact
            # we prefix the trajectory with the first 15 frames duplicated
            # and then cut them off after
            total_x = pandas.concat([prefix_x, (self.features[(part, 'x')])], axis = 0)
            total_y = pandas.concat([prefix_y, (self.features[(part, 'y')])], axis = 0)
            filtered_x = sig.sosfilt(sos, total_x)
            filtered_y = sig.sosfilt(sos, total_y)
            self.features[(part, 'x')] = filtered_x[15:]
            self.features[(part, 'y')] = filtered_y[15:]