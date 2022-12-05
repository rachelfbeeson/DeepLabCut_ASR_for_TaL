"""
This module partly uses code written by Sam Ribiero in the TaL Tools git repository
and Aciel Eshky in the UltraSuite Tools repository.

write_images_to_disk and create_video are modified versions of the same functions from animate_utterance.py
in the UltraSuite Tools repository. They have been modified to support both the ultrasound and video data. create_video
also now supports FFMPEG's GPU functionality.

video_handler is a combination of the animate_utterance function of the very same module,
and of visualiser.py from the Tal Tools repository.

Downsampling and trimming has been modified from the original tools.utils to support downsampling before
trimming. The dependencies tools.io and ustools.transform_ultrasound have been reproduced
without modification.
"""

import os
import shutil
import subprocess
import torch
import matplotlib.pyplot as plt
from tools import utils
from tools import io as myio
from tools.transform_ultrasound import transform_ultrasound
from tools.config_manager import config


class VideoMaker:
    """ Object to handle the creation of the tongue and lip videos.
    Since the lip and US videos are easy to handle by DLC if they are in one spot,
    the object is instantiated with those paths.
    """
    def __init__(self, us_output_path, lip_output_path):
        self.target_fps = config.getint('PreDLC', 'fps')
        self.us_output_path = us_output_path
        self.lip_output_path = lip_output_path
        self.param_temp = None
        self.meta_temp = None
        self.ult_temp = None
        self.vid_temp = None
        self.wav_temp = None
        self.wav_sr_temp = None
        self.temp_directory = '.temp'

    def write_images_to_disk(self, frames, origin):
        """
        A function to write the frames as images to a directory. The images are generated as plots without axes.
        :param frames: video frame data as a 3d numpy array
        :param origin: direction of the matplotlib axes (ultrasound and lip video use different orientations)
        """
        print("creating temporary directory...")
        if os.path.exists(self.temp_directory):
            shutil.rmtree(self.temp_directory)

        os.makedirs(self.temp_directory)

        print("writing image frames to disk...")
        plt.figure(dpi=300, figsize=((32 / 30), 0.8))

        c = frames[0]
        im = plt.imshow(c, aspect='auto', origin=origin, cmap='gray')
        for i in range(1, frames.shape[0]):
            c = frames[i]
            im.set_data(c)
            plt.axis("off")
            plt.savefig(self.temp_directory + "/%07d.jpg" % i, transparent=True, facecolor='black')

    def create_video(self, output_video_file):
        """
        A function to animate video frames. Frames are drawn and deposited in a temp
        directory, which are then used to animate in FFMPEG.
        Can use GPU to speed up if available.
        : param output_video_file: Where the video is saved
        """

        # check for gpu
        # we can speed up ffmpeg encoding by using a gpu, but quality is the tradeoff

        fps = str(self.target_fps)

        if torch.cuda.is_available():
            subprocess_list = ["ffmpeg", '-hwaccel_output_format', 'cuda', '-hwaccel', 'cuda', "-y", '-r', fps,
                               "-i", self.temp_directory + "/%07d.jpg", '-qp', '5', '-c:v', 'hevc_nvenc', '-r', fps,
                               output_video_file]
        else:
            subprocess_list = ["ffmpeg", "-y", '-r', fps,
                               "-i", self.temp_directory + "/%07d.jpg", '-crf', '10', '-r', fps,
                               output_video_file]

        subprocess.call(subprocess_list)
        print("Video saved.")

    def video_handler(self, utt_list):
        """
        Takes each utterance in the candidate list and prepares the videos for processing by DLC.
        The lip video and tongue ultrasound data are downsampled and trimmed, and the ultrasound data is transformed.
        The data are then turned into videos and saved in the format utt_id.mp4.
        Wav is also trimmed but not conserved - a future experimenter may want to use wav.
        @param candidate_list: A list of utterance objects.
        """

        if not os.path.isdir(self.us_output_path):
            os.mkdir(self.us_output_path)
        if not os.path.isdir(self.lip_output_path):
            os.mkdir(self.lip_output_path)

        for utt in utt_list:
            base_path = utt.base_path
            self.wav_temp, self.wav_sr_temp = myio.read_waveform(base_path + '.wav')
            self.ult_temp, self.param_temp = myio.read_ultrasound_tuple(base_path, shape='3d', cast=None, truncate=None)
            self.vid_temp, self.meta_temp = myio.read_video(base_path, shape='3d', cast=None)

            self.downsample()
            self.trim_to_parallel_streams()
            self.manipulate_ultrasound()

            print("Creating tongue video...")
            self.write_images_to_disk(self.ult_temp, origin='lower')
            self.create_video(os.path.join(self.us_output_path, f"{utt.id}.mp4"))
            print("Creating lip video...")
            self.write_images_to_disk(self.vid_temp, origin='upper')
            self.create_video(os.path.join(self.lip_output_path, f"{utt.id}.mp4"))

    def manipulate_ultrasound(self):
        """ Transforms ultrasound from US data into image data, then trimmed """
        ult_3d = self.ult_temp.reshape(-1, int(self.param_temp['NumVectors']), int(self.param_temp['PixPerVector']))
        ult_data = transform_ultrasound(ult_3d, background_colour=0, num_scanlines=int(self.param_temp['NumVectors']),
                                        size_scanline=int(self.param_temp['PixPerVector']),
                                        angle=float(self.param_temp['Angle']),
                                        zero_offset=int(self.param_temp['ZeroOffset']), pixels_per_mm=3)
        self.ult_temp = ult_data.transpose(0, 2, 1)[:, 0:650, 60:575]  # removes thick black border around US for DLC

    def trim_to_parallel_streams(self):
        ''' trim data to parallel streams '''

        vid_len = self.vid_temp.shape[0] / self.target_fps
        wav_len = self.wav_temp.shape[0] / self.wav_sr_temp

        # trim data streams to common start and end time stamps
        # ultrasound is always the last to start recording,
        # so we take that as start time
        start_time = self.param_temp['TimeInSecsOfFirstFrame']

        # video and audio finish recording first
        end_time = min(vid_len, wav_len)

        self.duration_temp = end_time - start_time
        # video
        import math
        frame_start = math.ceil(start_time * self.target_fps)
        frame_end = math.floor(end_time * self.target_fps)
        self.vid_temp = self.vid_temp[frame_start:frame_end, :, :]

        # audio
        sample_start = int(start_time * self.wav_sr_temp)
        sample_end = int(end_time * self.wav_sr_temp)
        self.wav_temp = self.vid_temp[sample_start:sample_end, ]

        # ultrasound

        frame_end = frame_end - frame_start
        self.ult_temp = self.ult_temp[:frame_end, :, :]

    def downsample(self):
        '''downsample ultrasound/video to target fps '''
        # resize ultrasound
        ultra_fps = self.param_temp['FramesPerSec']
        video_fps = self.meta_temp['fps']

        target_ult_frames = int(self.ult_temp.shape[0] * self.target_fps / ultra_fps)
        self.ult_temp = utils.resize(self.ult_temp, target_ult_frames)

        # resize video
        target_vid_frames = int(self.vid_temp.shape[0] * self.target_fps / video_fps)
        self.vid_temp = utils.resize(self.vid_temp, target_vid_frames)
