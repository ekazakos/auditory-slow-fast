
class AudioRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def participant(self):
        return NotImplementedError()

    @property
    def untrimmed_video_name(self):
        return NotImplementedError()

    @property
    def start_audio_sample(self):
        return NotImplementedError()

    @property
    def end_audio_sample(self):
        return NotImplementedError()

    @property
    def num_audio_samples(self):
        return NotImplementedError()

    @property
    def label(self):
        return NotImplementedError()

    @property
    def metadata(self):
        return NotImplementedError()
