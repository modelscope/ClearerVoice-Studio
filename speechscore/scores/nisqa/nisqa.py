from basis import ScoreBasis
from scores.nisqa.cal_nisqa import load_nisqa_model
import os

class NISQA(ScoreBasis):
    def __init__(self):
        super(NISQA, self).__init__(name='NISQA')
        self.intrusive = False
        self.score_rate = 48000
        self.model = load_nisqa_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights/nisqa.tar"), device='cpu')
 
    def windowed_scoring(self, audios, score_rate):
        from scores.nisqa.cal_nisqa import cal_NISQA
        score = cal_NISQA(self.model, audios[0])
        return score
