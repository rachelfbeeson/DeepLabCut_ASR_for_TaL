import os
import re
import nltk
from tools.config_manager import config

class KaldiFileMaker:
    """
    There are many files which are prescribed in order to run Kaldi, and this object
    creates those files. It builds the files one split at a time while maintaining
    files which are relevant for the entire data set.
    """
    def __init__(self):
        self.corpus = []
        self.lexicon = []
        self.dict = {}
        self.beep = config.getboolean('PostDLC', 'beep')
        self.s2g = []
        self.wav = []
        self.text = []
        self.u2s = []
        self.feats = []
        self.frame_shift = 1 / config.getint('PreDLC', 'fps')
        self.data_dir = "data"
        self.local_dir = os.path.join("data", "local")
        self.dict_dir = os.path.join(self.local_dir, "dict")
        # Non-silence phones covers all possible phones in CMUdict or BEEP dict
        # This would need to be changed if using a different lexicon with different phones
        self.non_silence_phones = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd',
                              'dh', 'ea', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ia', 'ih', 'iy',
                              'jh', 'k', 'l', 'm', 'n', 'ng', 'oh', 'ow', 'oy', 'p', 'r',
                              's', 'sh', 't', 'th', 'ua', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh']

    def make_dirs(self):
        """ Makes dirs which Kaldi scripts will rely on """
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)
        if not os.path.isdir(self.local_dir):
            os.mkdir(self.local_dir)
        if not os.path.isdir(self.dict_dir):
            os.mkdir(self.dict_dir)

    def make_language_files(self):
        """
        Makes and writes the language files for an entire data set which are
        prescribed by Kaldi, like the lexicon. Meant to be called after all
        splits have been seen by the object, so the corpus contains text from every split.
        """
        self.lexicon_maker()
        self.file_writer(self.corpus, self.local_dir, 'corpus.txt')
        self.file_writer(self.lexicon, self.dict_dir, 'lexicon.txt')
        self.file_writer(['sil\n'], self.dict_dir, 'optional_silence.txt')
        self.file_writer(['sil\n', 'spn\n'], self.dict_dir, 'silence_phones.txt')
        self.file_writer([a.upper() + '\n' for a in self.non_silence_phones], self.dict_dir, 'nonsilence_phones.txt')

    def lexicon_maker(self):
        """
        Builds the lexicon which contains the phone sequence corresponding to the words in the corpus.
        """
        self.get_pronunciation_dict()
        self.lexicon = ['!SIL sil\n', '<UNK> spn\n']
        not_in = []
        for phrase in self.corpus:
            phrase = re.sub(r'[^\w|\s|\-|\:|\'|\;|]', '', phrase)
            phrase = phrase.strip('\n')
            phrase = phrase.split(' ')
            for word in phrase:
                if self.beep:
                    word = word.upper()
                else:
                    word = word.lower()
                try:
                    list_word_phone_seq = self.dict[word]
                except KeyError:
                    print(f'The word {word} is not in the dictionary.')
                    print(f'Ignoring {word}...')
                    not_in.append(word)  # can use this to manually add lexicon entries if needed
                    continue
                for pronun in list_word_phone_seq:
                    if self.beep:
                        self.lexicon.append(word.lower() + ' ' + pronun.upper() + '\n')
                    else:
                        pronun = ' '.join(pronun)
                        pronun = re.sub(r'[\d]', '', pronun)
                        self.lexicon.append(word + ' ' + pronun.upper() + '\n')
        self.lexicon = list(set(self.lexicon))
        self.lexicon.sort()

    def get_pronunciation_dict(self):
        """ Builds the BEEP dict if using, otherwise just nabs CMUdict from nltk """
        if self.beep:
            pronunciation_text = config.get('Paths', 'beep_path')
            with open(pronunciation_text, 'r') as f:
                dict_lines = f.readlines()
            for line in dict_lines:
                if line[0] == '#':
                    continue
                if '\t' in line:
                    line = line.split('\t', 1)
                else:
                    line = line.split(' ', 1)
                pron = line[1].strip('\t')
                pron = pron.strip('\n')
                self.dict[line[0]] = [pron]
        else:
            self.dict = nltk.corpus.cmudict.dict()

    def make_kaldi_files(self, utts, split):
        """
        Creates the files specific to the data splits, in addition to the corpus of the entire data set.
        speak2gender, wav.scp, utt2spk, and text.
        The contents of these files are prescribed by Kaldi.
        @param utts: List of utterance objects from one split
        @param split: string for which split the utt belongs to
        """

        for utt in utts:
            if utt.discarded:
                pass
            self.s2g.append(utt.speaker + ' ' + utt.gender + '\n')
            self.s2g = list(set(self.s2g))
            wav_path = utt.base_path + '.wav'
            self.wav.append(utt.id + ' ' + wav_path + '\n')
            text_content = re.sub(r'[^\w|\s|\']', '', utt.text)
            text_content = text_content.lower()
            self.text.append(utt.id + ' ' + text_content)
            self.corpus.append(text_content)
            self.u2s.append(utt.id + ' ' + utt.speaker + '\n')
            self.kaldi_features(utt)
        self.s2g.sort()
        self.wav.sort()
        self.text.sort()
        self.u2s.sort()
        self.write_files(split)

        # prepare for next split
        self.s2g = []
        self.wav = []
        self.text = []
        self.u2s = []
        self.feats = []

    def write_files(self, split):
        """ Writes files which are specific to a data split """
        path = os.path.join("data", split)
        if not os.path.isdir(path):
            os.mkdir(path)

        # not using wavs for this experiment so not making wav.scp

        files_to_write = [self.s2g, self.text, self.u2s, self.feats, str(self.frame_shift)]
        file_names = ['spk2gender', 'text', 'utt2spk', 'feats.txt', 'frame_shift']
        for file_content, name in zip(files_to_write, file_names):
            self.file_writer(file_content, path, name)

    def kaldi_features(self, utt):
        """
        Turns the numpy matrices within the feature dict into a text format which Kaldi expects.
        I.e. utt_id [ 0 1 .001 .2 .03 0... ]
        @param utt: utterance object
        """
        feats = ''
        feats += utt.id + ' [ '
        first = True
        for row in utt.combined_feats:
            row_string = ' '.join(str(e) for e in row)
            if first == True:
                feats += row_string
                first = False
            else:
                feats += '\n' + row_string
        feats += ' ]'
        feats += '\n'
        self.feats += feats

    @staticmethod
    def file_writer(file_content, path, file_name):
        """
        Writes the file content to the path with the specified name.
        """
        with open(os.path.join(path, file_name), 'w') as f:
            for line in file_content:
                f.write(line)

