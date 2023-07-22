from nnmnkwii.datasets import FileDataSource
from collections import OrderedDict
from glob import glob
from os.path import basename, exists, join, splitext
import numpy as np

feamle_speakers = ['06869', '05705', '05994', '08664', '05691', '08318', '04830', '09935', '03910', '08712', '07789', '02828',
                       '06928', '01033', '04261', '04213', '06625', '01908', '06566', '06008', '05181', '05484', '09281', '03689',
]

male_spkrs = ['07220', '06359', '02812', '08680', '02769', '04272', '00682', '09525', '05425', '03338', '00974', '04771', 
                       '00113', '08377', '09222', '05479', '09015', '09584', '06885', '06582', '06917', '02477', '07486']

available_speakers = feamle_speakers + male_spkrs



def _parse_speaker_info(data_root):
    speaker_info = OrderedDict()
    for spkr in feamle_speakers:
        speaker_info[spkr] = {}
        speaker_info[spkr]["gender"] = "Female"
    for spkr in male_spkrs:
        speaker_info[spkr] = {}
        speaker_info[spkr]["gender"] = "Male"
    return speaker_info
        


class TeluguBaseDataSource(FileDataSource,):
    def __init__(self, data_root, speakers, labelmap, max_files) -> None:
        self.data_root = data_root
        if speakers == "all":
            speakers = available_speakers
        for speaker in speakers:
            if speaker not in available_speakers:
                raise ValueError(
                    "Unknown speaker '{}'. It should be one of {}".format(
                    speaker, available_speakers
                )
            )
        self.speakers = speakers
        if labelmap is None:
            labelmap = {}
            for idx, speaker in enumerate(speakers):
                labelmap[speaker] = idx

        self.labelmap = labelmap
        self.labels = None
        self.max_files = max_files

        self.speaker_info = _parse_speaker_info(data_root)



    def _validate(self):
        # should have pair of transcription and wav files
        for _, speaker in enumerate(self.speakers):
            txt_files = sorted(
                glob(
                    join(
                        self.data_root,
                        "*_{}_*.txt".format(speaker),
                    )
                )
            )
            wav_files = sorted(
                glob(
                    join(
                        self.data_root,
                        "*_{}_*.wav".format(speaker),
                    )
                )
            )
            assert len(txt_files) > 0
            for txt_path, wav_path in zip(txt_files, wav_files):
                assert (
                    splitext(basename(txt_path))[0] == splitext(basename(wav_path))[0]
                )
    
    def collect_files(self, is_wav):
        if is_wav:
            ext = ".wav"
        else:
            ext = ".txt"

        paths = []
        labels = []
        if self.max_files is None:
            max_files_per_speaker = None
        else:
            max_files_per_speaker = self.max_files // len(self.speakers)
            print("test",max_files_per_speaker)
        for idx, speaker in enumerate(self.speakers):
            files = sorted(glob(join(self.data_root,"te*_{}_*{}".format(speaker, ext))))
            files = files[:max_files_per_speaker]
            if not is_wav:
                files = list(
                    map(lambda s: open(s, "rb").read().decode("utf-8")[:-1], files)
                )
            for f in files:
                paths.append(f)
                labels.append(self.labelmap[self.speakers[idx]])
        self.labels = np.array(labels, dtype=np.int16)

        return paths

class TranscriptionDataSource(TeluguBaseDataSource):
    """Transcription data source for JVS dataset

    The data source collects text transcriptions from JVS.
    Users are expected to inherit the class and implement ``collect_features``
    method, which defines how features are computed given a transcription.

    Args:
        data_root (str): Data root.
        speakers (list): List of speakers to find. Speaker id must be ``str``.
          For supported names of speaker, please refer to ``available_speakers``
          defined in the module.
        labelmap (dict[optional]): Dict of speaker labels. If None,
          it's assigned as incrementally (i.e., 0, 1, 2) for specified
          speakers.
        max_files (int): Total number of files to be collected.

    Attributes:
        speaker_info (dict): Dict of speaker information dict. Keyes are speaker
          ids (str) and each value is speaker information consists of ``gender``,
          ``minf0`` and ``maxf0``.
        labels (numpy.ndarray): Speaker labels paired with collected files.
          Stored in ``collect_files``. This is useful to build multi-speaker
          models.

    """
    def __init__(self, data_root, speakers = available_speakers, labelmap = None, max_files = None):
        super(TranscriptionDataSource, self).__init__(data_root, speakers, labelmap, max_files)
    
    def collect_files(self, nonpara = False, whisper = False):
        return super(TranscriptionDataSource, self).collect_files(False)

class WavFileDataSource(TeluguBaseDataSource):
    """WavFile data source for JVS dataset.

    The data source collects text transcriptions from JVS.
    Users are expected to inherit the class and implement ``collect_features``
    method, which defines how features are computed given a transcription.

    Args:
        data_root (str): Data root.
        speakers (list): List of speakers to find. Speaker id must be ``str``.
          For supported names of speaker, please refer to ``available_speakers``
          defined in the module.
        labelmap (dict[optional]): Dict of speaker labels. If None,
          it's assigned as incrementally (i.e., 0, 1, 2) for specified
          speakers.
        max_files (int): Total number of files to be collected.

    Attributes:
        speaker_info (dict): Dict of speaker information dict. Keyes are speaker
          ids (str) and each value is speaker information consists of ``gender``,
          ``minf0`` and ``maxf0``.
        labels (numpy.ndarray): Speaker labels paired with collected files.
          Stored in ``collect_files``. This is useful to build multi-speaker
          models.
    """
    def __init__(self, data_root, speakers = available_speakers, labelmap = None, max_files = None):
        super(WavFileDataSource, self).__init__(data_root, speakers, labelmap, max_files)
    
    def collect_files(self, nonpara = False, whisper = False):
        return super(WavFileDataSource, self).collect_files(True)




if __name__ == "__main__":
    source = TeluguBaseDataSource("/home/lohith/telugu_dataset","all",None,None)
    paths = source.collect_files(False)
    print(paths)