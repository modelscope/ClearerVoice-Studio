from basis import ScoreBasis
from scores.distill_mos.sqa import ConvTransformerSQAModel
import numpy as np
import torch

class DISTILL_MOS(ScoreBasis):
    def __init__(self):
        super(DISTILL_MOS, self).__init__(name='DISTILL_MOS')
        self.intrusive = False
        self.score_rate = 16000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConvTransformerSQAModel().to(self.device)
        self.model.eval()

    def windowed_scoring(self, audios, score_rate):
    	
        score = self.model(torch.from_numpy(np.expand_dims(audios[0], axis=0)).float().to(self.device))
        score_np = score.detach().cpu().numpy()
        return score_np[0][0]