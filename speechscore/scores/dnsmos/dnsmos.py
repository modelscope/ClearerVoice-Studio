import os

import librosa
import numpy as np
import numpy.polynomial.polynomial as poly
import onnxruntime as ort
import soundfile as sf

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

from basis import ScoreBasis

module_dir = os.path.dirname(os.path.abspath(__file__))

class DNSMOS(ScoreBasis):
    def __init__(self):
        super(DNSMOS, self).__init__(name='DNSMOS')
        self.intrusive = True
        self.score_rate = 16000
        self.p808_model_path = os.path.join(module_dir, 'DNSMOS', 'model_v8.onnx')
        self.primary_model_path = os.path.join(module_dir, 'DNSMOS', 'sig_bak_ovr.onnx')
        self.compute_score = ComputeScore(self.primary_model_path, self.p808_model_path)

    def windowed_scoring(self, audios, rate):
        return self.compute_score.cal_mos(audios[0], rate)

class ComputeScore:
    def __init__(self, primary_model_path, p808_model_path) -> None:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.onnx_sess = ort.InferenceSession(primary_model_path, providers=providers)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path, providers=providers)
        
    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max)+40)/40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr):
        p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
        p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
        p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def cal_mos(self, audio, sampling_rate):
        fs = sampling_rate
        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH*fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        
        num_hops = int(np.floor(len(audio)/fs) - INPUT_LENGTH)+1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx*hop_len_samples) : int((idx+INPUT_LENGTH)*hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis,:]
            p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis, :, :]
            oi = {'input_1': input_features}
            p808_oi = {'input_1': p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw,mos_bak_raw,mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig,mos_bak,mos_ovr = self.get_polyfit_val(mos_sig_raw,mos_bak_raw,mos_ovr_raw)
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        results = {}
        results['OVRL'] = np.mean(predicted_mos_ovr_seg)
        results['SIG'] = np.mean(predicted_mos_sig_seg)
        results['BAK'] = np.mean(predicted_mos_bak_seg)
        results['P808_MOS'] = np.mean(predicted_p808_mos)
        return results
