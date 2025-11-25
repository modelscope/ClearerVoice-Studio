class ScoreBasis:
    def __init__(self, name=None):
        # the score operates on the specified rate
        self.score_rate = None
        # is the score intrusive or non-intrusive ?
        self.intrusive = True #require a reference
        self.name = name
        self.model = None
        self.device = 'cpu'

    def windowed_scoring(self, audios, score_rate):
        raise NotImplementedError(f'In {self.name}, windowed_scoring is not yet implemented')

    def scoring(self, data, window=None, score_rate=None):
        """ calling the `windowed_scoring` function that should be specialised
        depending on the score."""

        # imports
        #import soundfile as sf
        import resampy
        from museval.metrics import Framing

        #checking rate
        audios = data['audio'].copy()
        score_rate = data['rate']

        if self.score_rate is not None:
            score_rate = self.score_rate

        if score_rate != data['rate']:
            for index, audio in enumerate(audios):
                audio = resampy.resample(audio, data['rate'], score_rate, axis=0)
                audios[index] = audio

        if window is not None:
            maxlen = len(audios[0])
            framer = Framing(window * score_rate, window * score_rate, maxlen)
            nwin = framer.nwin
            result = {}
            for (t, win) in enumerate(framer):
                result_t = self.windowed_scoring([audio[win] for audio in audios], score_rate)
                result[t] = result_t
            if win and maxlen > win.stop:
                last_win = slice(win.stop, maxlen)
                result_t = self.windowed_scoring([audio[last_win] for audio in audios], score_rate) 
                result[nwin] = result_t
        else:
            result = self.windowed_scoring(audios, score_rate)
        return result
