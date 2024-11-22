# speechscore

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Usage](#2-usage)

## 1. Introduction

speechscore is a wrapper designed for assessing speech quality. It includes a collection of commonly used speech quality metrics, as listed below:
| Index | Metrics | Short Description | Externel Link |
|-------|---------|-------------|---------------|
|1.| BSSEval {ISR, SAR, SDR} | | (See <a href="https://github.com/sigsep/sigsep-mus-eval">the official museval page</a>)|
|2.| {CBAK, COVL, CSIG} | CSIG predicts the signal distortion mean opinion score (MOS), CBAK measures background intrusiveness, and COVL measures speech quality. CSIG, CBAK, and COVL are ranged from 1 to 5| See paper: <a href="https://ecs.utdallas.edu/loizou/speech/obj_paper_jan08.pdf">Evaluation of Objective Quality Measures for Speech Enhancement</a>|
|3.| DNSMOS {BAK, OVRL, SIG, P808_MOS} |||
|4.| FWSEGSNR | | | |
|5.| LLR | | |
|6.| LSD | | |
|7.| MCD | | |
|8.| NB_PESQ | | |
|9.| PESQ | | |
|10.| SISDR | | |
|11.| SNR | | |
|12.| SRMR | | |
|13.| SSNR | | |
|14.| STOI| | |

## 2. Usage

### Step-by-Step Guide

1. **Clone the Repository**

``` sh
git clone https://github.com/modelscope/ClearerVoice-Studio.git
```

2. **Create Conda Environment**

If you haven't created conda envrionment for ClearerVoice, follow the instructions below:

``` sh
cd ClearerVoice-Studio
conda create -n ClearerVoice python=3.8
conda activate ClearerVoice
pip install -r requirements.txt
```

3. Run demo script

``` sh
cd speechscore
python demo.py
```
or use the following script:
``` python
# Import pprint for pretty-printing the results in a more readable format
import pprint
# Import the SpeechScore class to evaluate speech quality metrics
from speechscore import SpeechScore 

# Main block to ensure the code runs only when executed directly
if __name__ == '__main__':
    # Initialize a SpeechScore object with a list of score metrics to be evaluated
    # Supports any subsets of the list
    mySpeechScore = SpeechScore([
        'SRMR', 'PESQ', 'NB_PESQ', 'STOI', 'SISDR', 
        'FWSEGSNR', 'LSD', 'BSSEval', 'DNSMOS', 
        'SNR', 'SSNR', 'LLR', 'CSIG', 'CBAK', 
        'COVL', 'MCD'
    ])

    # Call the SpeechScore object to evaluate the speech metrics between 'noisy' and 'clean' audio
    # Arguments:
    # - {test_path, reference_path} supports audio directories or audio paths (.wav or .flac)
    # - window (float): seconds, set None to specify no windowing (process the full audio)
    # - score_rate (int): specifies the sampling rate at which the metrics should be computed
    # - return_mean (bool): set True to specify that the mean score for each metric should be returned

    
    print('score for a signle wav file')
    scores = mySpeechScore(test_path='audios/noisy.wav', reference_path='audios/clean.wav', window=None, score_rate=16000, return_mean=False)
    # Pretty-print the resulting scores in a readable format
    pprint.pprint(scores)

    print('score for wav directories')
    scores = mySpeechScore(test_path='audios/noisy/', reference_path='audios/clean/', window=None, score_rate=16000, return_mean=True)

    # Pretty-print the resulting scores in a readable format
    pprint.pprint(scores)

    # Print only the resulting mean scores in a readable format
    #pprint.pprint(scores['Mean_Score'])
```
The results should be looking like below:

```sh
score for a signle wav file
{'BSSEval': {'ISR': 22.74466768594831,
             'SAR': -0.1921607960486258,
             'SDR': -0.23921670199308115},
 'CBAK': 1.5908301020179343,
 'COVL': 1.5702204013203889,
 'CSIG': 2.3259366746377066,
 'DNSMOS': {'BAK': 1.3532928733331306,
            'OVRL': 1.3714771994335782,
            'P808_MOS': 2.354834,
            'SIG': 1.8698058813241407},
 'FWSEGSNR': 6.414399025759913,
 'LLR': 0.85330075,
 'LSD': 2.136734818644327,
 'MCD': 11.013451521306235,
 'NB_PESQ': 1.2447538375854492,
 'PESQ': 1.0545592308044434,
 'SISDR': -0.23707451176264824,
 'SNR': -0.9504614142497447,
 'SRMR': 6.202590182397157,
 'SSNR': -0.6363067113236048,
 'STOI': 0.8003376411051097}
 
score for wav directories
{'Mean_Score': {'BSSEval': {'ISR': 23.728811184378372,
                            'SAR': 4.839625092004951,
                            'SDR': 4.9270216975279135},
                'CBAK': 1.9391528046230797,
                'COVL': 1.5400270840455588,
                'CSIG': 2.1286157747587344,
                'DNSMOS': {'BAK': 1.9004402577440938,
                           'OVRL': 1.860621534493506,
                           'P808_MOS': 2.5821499824523926,
                           'SIG': 2.679913397827385},
                'FWSEGSNR': 9.079539440199582,
                'LLR': 1.1992616951465607,
                'LSD': 2.0045290996104748,
                'MCD': 8.916492705343465,
                'NB_PESQ': 1.431145429611206,
                'PESQ': 1.141619324684143,
                'SISDR': 4.778657656271212,
                'SNR': 4.571920494312266,
                'SRMR': 9.221118316293268,
                'SSNR': 2.9965604574762796,
                'STOI': 0.8585249663711918},
 'audio_1.wav': {'BSSEval': {'ISR': 22.74466768594831,
                             'SAR': -0.1921607960486258,
                             'SDR': -0.23921670199308115},
                 'CBAK': 1.5908301020179343,
                 'COVL': 1.5702204013203889,
                 'CSIG': 2.3259366746377066,
                 'DNSMOS': {'BAK': 1.3532928733331306,
                            'OVRL': 1.3714771994335782,
                            'P808_MOS': 2.354834,
                            'SIG': 1.8698058813241407},
                 'FWSEGSNR': 6.414399025759913,
                 'LLR': 0.85330075,
                 'LSD': 2.136734818644327,
                 'MCD': 11.013451521306235,
                 'NB_PESQ': 1.2447538375854492,
                 'PESQ': 1.0545592308044434,
                 'SISDR': -0.23707451176264824,
                 'SNR': -0.9504614142497447,
                 'SRMR': 6.202590182397157,
                 'SSNR': -0.6363067113236048,
                 'STOI': 0.8003376411051097},
 'audio_2.wav': {'BSSEval': {'ISR': 24.712954682808437,
                             'SAR': 9.871410980058528,
                             'SDR': 10.093260097048908},
                 'CBAK': 2.287475507228225,
                 'COVL': 1.509833766770729,
                 'CSIG': 1.9312948748797627,
                 'DNSMOS': {'BAK': 2.4475876421550566,
                            'OVRL': 2.349765869553434,
                            'P808_MOS': 2.809466,
                            'SIG': 3.490020914330629},
                 'FWSEGSNR': 11.744679854639253,
                 'LLR': 1.5452226,
                 'LSD': 1.8723233805766222,
                 'MCD': 6.819533889380694,
                 'NB_PESQ': 1.617537021636963,
                 'PESQ': 1.2286794185638428,
                 'SISDR': 9.794389824305073,
                 'SNR': 10.094302402874277,
                 'SRMR': 12.23964645018938,
                 'SSNR': 6.629427626276164,
                 'STOI': 0.9167122916372739}}
```
Any subset of the full score list is supported, specify your score list using the following objective:

```
mySpeechScore = SpeechScore(['.'])
```



