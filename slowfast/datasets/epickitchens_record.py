from .audio_record import AudioRecord
from datetime import timedelta
import time


def timestamp_to_sec(timestamp):
    x = time.strptime(timestamp, '%H:%M:%S.%f')
    sec = float(timedelta(hours=x.tm_hour,
                          minutes=x.tm_min,
                          seconds=x.tm_sec).total_seconds()) + float(
        timestamp.split('.')[-1]) / 100
    return sec


class EpicKitchensAudioRecord(AudioRecord):
    def __init__(self, tup):
        self._index = str(tup[0])
        self._series = tup[1]

    @property
    def participant(self):
        return self._series['participant_id']

    @property
    def untrimmed_video_name(self):
        return self._series['video_id']

    @property
    def start_audio_sample(self):
        return int(round(timestamp_to_sec(self._series['start_timestamp']) * 24000))

    @property
    def end_audio_sample(self):
        return int(round(timestamp_to_sec(self._series['stop_timestamp']) * 24000))

    @property
    def num_audio_samples(self):
        return self.end_audio_sample - self.start_audio_sample

    @property
    def label(self):
        return {'verb': self._series['verb_class'] if 'verb_class' in self._series else -1,
                'noun': self._series['noun_class'] if 'noun_class' in self._series else -1}

    @property
    def metadata(self):
        return {'narration_id': self._index}
