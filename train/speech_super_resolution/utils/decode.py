
#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
#Author: Shengkui Zhao

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch 
import torch.nn as nn
import numpy as np
import os 
import sys
import librosa
import torchaudio
from utils.misc import power_compress, power_uncompress, stft, istft, compute_fbank
from utils.bandwidth_sub import bandwidth_sub
from dataloader.meldataset import mel_spectrogram

MAX_WAV_VALUE = 32768.0

def decode_one_audio(model, device, inputs, args):
    if args.network == 'FRCRN_SE_16K':
        return decode_one_audio_frcrn_se_16k(model, device, inputs, args)
    elif args.network == 'MossFormer2_SE_48K':
        return decode_one_audio_mossformer2_se_48k(model, device, inputs, args)
    elif args.network == 'MossFormerGAN_SE_16K':
        return decode_one_audio_mossformergan_se_16k(model, device, inputs, args)
    elif args.network == 'MossFormer2_SR_48K':
        return decode_one_audio_mossformer2_sr_48k(model, device, inputs, args)
    else:
       print("No network found!")
       return 

def get_mel(x, args):
    """
    Calls mel_spectrogram() and returns the mel-spectrogram output
    """
    
    return mel_spectrogram(x, args.n_fft, args.num_mels, args.sampling_rate, args.hop_size, args.win_size, args.fmin, args.fmax)
    
def decode_one_audio_mossformer2_sr_48k(model, device, inputs, args):
    """
    This function decodes a single audio input using a two-stage speech super-resolution model.
    Supports both offline decoding (for short audio) and online decoding (for long audio)
    with a sliding window approach.

    Parameters:
    -----------
    model : list
        A list of two-stage models:
        - model[0]: The transformer-based Mossformer model for feature enhancement.
        - model[1]: The vocoder for generating high-resolution waveforms.
    device : str or torch.device
        The computation device ('cpu' or 'cuda') where the models will run.
    inputs : torch.Tensor
        A tensor of shape (batch_size, num_samples) containing low-resolution audio signals.
        Only the first audio (inputs[0, :]) is processed.
    args : Namespace
        An object containing the following attributes:
        - sampling_rate: Sampling rate of the input audio (e.g., 48,000 Hz).
        - one_time_decode_length: Maximum duration (in seconds) for offline decoding.
        - decode_window: Window size (in seconds) for sliding window processing.
        - Other optional attributes used for Mel spectrogram extraction.

    Returns:
    --------
    numpy.ndarray
        The high-resolution audio waveform as a NumPy array, refined and upsampled.
    """
    inputs = inputs[0, :]  # Extract the first element from the input tensor
    input_len = inputs.shape[0]  # Get the length of the input audio
    #inputs = inputs * MAX_WAV_VALUE  # Normalize the input to the maximum WAV value

    # Check if input length exceeds the defined threshold for online decoding
    if input_len > args.sampling_rate * args.one_time_decode_length:  # 20 seconds
        online_decoding = True
        if online_decoding:
            window = int(args.sampling_rate * args.decode_window)  # Define window length (e.g., 4s for 48kHz)
            stride = int(window * 0.75)  # Define stride length (e.g., 3s for 48kHz)
            t = inputs.shape[0]  # Update length after potential padding

            # Pad input if necessary to match window size
            if t < window:
                inputs = np.concatenate([inputs, np.zeros(window - t)], 0)
            elif t < window + stride:
                padding = window + stride - t
                inputs = np.concatenate([inputs, np.zeros(padding)], 0)
            else:
                if (t - window) % stride != 0:
                    padding = t - (t - window) // stride * stride
                    inputs = np.concatenate([inputs, np.zeros(padding)], 0)

            audio = torch.from_numpy(inputs).type(torch.FloatTensor)  # Convert to Torch tensor
            t = audio.shape[0]  # Update length after conversion
            outputs = torch.from_numpy(np.zeros(t))  # Initialize output tensor
            give_up_length = (window - stride) // 2  # Determine length to ignore at the edges
            dfsmn_memory_length = 0  # Placeholder for potential memory length
            current_idx = 0  # Initialize current index for sliding window

            # Process audio in sliding window segments
            while current_idx + window <= t:
                # Select appropriate segment of audio for processing
                if current_idx < dfsmn_memory_length:
                    audio_segment = audio[0:current_idx + window]
                else:
                    audio_segment = audio[current_idx - dfsmn_memory_length:current_idx + window]

                # Pass filter banks through the model
                mel_segment = get_mel(audio_segment.unsqueeze(0), args)
                mossformer_output_segment = model[0](mel_segment.to(device))
                generator_output_segment = model[1](mossformer_output_segment)
                generator_output_segment = generator_output_segment.squeeze()
                offset = len(audio_segment) - len(generator_output_segment)
                # Store the output segment in the output tensor
                if current_idx == 0:
                    outputs[current_idx:current_idx + window - give_up_length] = generator_output_segment[:-give_up_length+offset]
                else:
                    generator_output_segment = generator_output_segment[-window:]  # Get the latest window of output
                    outputs[current_idx + give_up_length:current_idx + window - give_up_length] = generator_output_segment[give_up_length:-give_up_length+offset]
                
                current_idx += stride  # Move to the next segment

    else:
        # Process the entire audio at once if it is shorter than the threshold
        audio = torch.from_numpy(inputs).type(torch.FloatTensor)
        mel_input = get_mel(audio.unsqueeze(0), args)
        mossformer_output = model[0](mel_input.to(device))
        generator_output = model[1](mossformer_output)
        outputs = generator_output.squeeze()

    outputs = outputs.cpu().numpy()
    outputs = bandwidth_sub(inputs, outputs)
    return outputs

def decode_one_audio_frcrn_se_16k(model, device, inputs, args):

    decode_do_segement=False

    window = args.sampling_rate * args.decode_window  #decoding window length 16000 # 1s
    stride = int(window*0.75)
    b,t = inputs.shape

    if t > args.sampling_rate * args.one_time_decode_length:
        decode_do_segement=True
    if t < window:
        inputs = np.concatenate([inputs,np.zeros((inputs.shape[0],window-t))],1)
    elif t < window + stride:
        padding = window + stride - t
        inputs = np.concatenate([inputs,np.zeros((inputs.shape[0],padding))],1)
    else:
        if (t - window) % stride != 0:
            padding = t - (t-window)//stride * stride
            inputs = np.concatenate([inputs,np.zeros((inputs.shape[0],padding))],1)

    inputs = torch.from_numpy(np.float32(inputs))
    inputs = inputs.to(device)
    b,t = inputs.shape
    if decode_do_segement:
        outputs = np.zeros(t)
        give_up_length=(window - stride)//2
        current_idx = 0
        while current_idx + window <= t:
            tmp_input = inputs[:,current_idx:current_idx+window]
            tmp_output = model.inference(tmp_input,).cpu().numpy()
            if current_idx == 0:
                outputs[current_idx:current_idx+window-give_up_length] = tmp_output[:-give_up_length]

            else:
                outputs[current_idx+give_up_length:current_idx+window-give_up_length] = tmp_output[give_up_length:-give_up_length]
            current_idx += stride
    else:
        outputs = model.inference(inputs,).cpu().numpy()

    return outputs

def decode_one_audio_mossformergan_se_16k(model, device, inputs, args):

    decode_do_segement=False
    window = args.sampling_rate * args.decode_window #16000 # 1s
    stride = int(window*0.75)
    b,t = inputs.shape
    if t > args.sampling_rate * args.one_time_decode_length:
        decode_do_segement=True

    if t < window:
        inputs = np.concatenate([inputs,np.zeros((inputs.shape[0],window-t))],1)
    elif t < window + stride:
        padding = window + stride - t
        inputs = np.concatenate([inputs,np.zeros((inputs.shape[0],padding))],1)
    else:
        if (t - window) % stride != 0:
            padding = t - (t-window)//stride * stride
            inputs = np.concatenate([inputs,np.zeros((inputs.shape[0],padding))],1)
    inputs = torch.from_numpy(np.float32(inputs))
    inputs = inputs.to(device)
    b,t = inputs.shape
    if decode_do_segement:
        outputs = np.zeros(t)
        give_up_length=(window - stride)//2
        current_idx = 0
        while current_idx + window <= t:
            tmp_input = inputs[:,current_idx:current_idx+window]
            tmp_output = _decode_one_audio_mossformergan_se_16k(model, device, tmp_input, args)
            if current_idx == 0:
                outputs[current_idx:current_idx+window-give_up_length] = tmp_output[:-give_up_length]

            else:
                outputs[current_idx+give_up_length:current_idx+window-give_up_length] = tmp_output[give_up_length:-give_up_length]
            current_idx += stride
        return outputs
    else:
        return _decode_one_audio_mossformergan_se_16k(model, device, inputs, args)

def _decode_one_audio_mossformergan_se_16k(model, device, inputs, args):
    input_len = inputs.size(-1)
    nframe = int(np.ceil(input_len / args.win_inc))
    padded_len = nframe * args.win_inc
    padding_len = padded_len - input_len
    inputs = torch.cat([inputs, inputs[:, :padding_len]], dim=-1)

    c = torch.sqrt(inputs.size(-1) / torch.sum((inputs ** 2.0), dim=-1))
    inputs = torch.transpose(inputs, 0, 1)
    inputs = torch.transpose(inputs * c, 0, 1)
    inputs_spec = stft(inputs, args, center=True)
    inputs_spec = inputs_spec.to(torch.float32)
    inputs_spec = power_compress(inputs_spec).permute(0, 1, 3, 2)
    out_list = model(inputs_spec)
    pred_real, pred_imag = out_list[0].permute(0, 1, 3, 2), out_list[1].permute(0, 1, 3, 2)
    pred_spec_uncompress = power_uncompress(pred_real, pred_imag).squeeze(1)
    outputs = istft(pred_spec_uncompress, args)
    outputs = outputs.squeeze(0) / c
    outputs = outputs[:input_len]
    return outputs.detach().cpu().numpy()

def decode_one_audio_mossformer2_se_48k(model, device, inputs, args):
    inputs = inputs[0,:]
    input_len = inputs.shape[0]
    inputs = inputs * MAX_WAV_VALUE
    if input_len > args.sampling_rate * args.one_time_decode_length: ## longer than 20s, use online decoding
        online_decoding = True
        if online_decoding:
            window = int(args.sampling_rate * args.decode_window) ## 4s window for 48kHz sample rate
            stride = int(window * 0.75) ## 3s stride for 48kHz sample rate
            t = inputs.shape[0]

            if t < window:
                inputs = np.concatenate([inputs,np.zeros(window-t)],0)
            elif t < window + stride:
                padding = window + stride - t
                inputs = np.concatenate([inputs,np.zeros(padding)],0)
            else:
                if (t - window) % stride != 0:
                    padding = t - (t-window)//stride * stride
                    inputs = np.concatenate([inputs,np.zeros(padding)],0)
            audio = torch.from_numpy(inputs).type(torch.FloatTensor)
            t = audio.shape[0]
            outputs = torch.from_numpy(np.zeros(t))
            give_up_length=(window - stride)//2
            dfsmn_memory_length = 0 
            current_idx = 0
            while current_idx + window <= t:
                if current_idx < dfsmn_memory_length:
                    audio_segment = audio[0:current_idx+window]
                else:
                    audio_segment = audio[current_idx-dfsmn_memory_length:current_idx+window]
                fbanks = compute_fbank(audio_segment.unsqueeze(0), args)
                # compute deltas for fbank
                fbank_tr = torch.transpose(fbanks, 0, 1)
                fbank_delta = torchaudio.functional.compute_deltas(fbank_tr)
                fbank_delta_delta = torchaudio.functional.compute_deltas(fbank_delta)
                fbank_delta = torch.transpose(fbank_delta, 0, 1)
                fbank_delta_delta = torch.transpose(fbank_delta_delta, 0, 1)
                fbanks = torch.cat([fbanks, fbank_delta, fbank_delta_delta], dim=1)

                fbanks =fbanks.unsqueeze(0).to(device)
                Out_List = model(fbanks)
                pred_mask = Out_List[-1]
                spectrum = stft(audio_segment, args)
                pred_mask = pred_mask.permute(2,1,0)
                masked_spec = spectrum.cpu() * pred_mask.detach().cpu()
                masked_spec_complex = masked_spec[:,:,0] + 1j*masked_spec[:,:,1]
                output_segment = istft(masked_spec_complex, args, len(audio_segment))
                if current_idx == 0:
                    outputs[current_idx:current_idx+window-give_up_length] = output_segment[:-give_up_length]
                else:
                    output_segment = output_segment[-window:]
                    outputs[current_idx+give_up_length:current_idx+window-give_up_length] = output_segment[give_up_length:-give_up_length]
                current_idx += stride
    else:
        audio = torch.from_numpy(inputs).type(torch.FloatTensor)
        fbanks = compute_fbank(audio.unsqueeze(0), args)
        # compute deltas for fbank
        fbank_tr = torch.transpose(fbanks, 0, 1)
        fbank_delta = torchaudio.functional.compute_deltas(fbank_tr)
        fbank_delta_delta = torchaudio.functional.compute_deltas(fbank_delta)
        fbank_delta = torch.transpose(fbank_delta, 0, 1)
        fbank_delta_delta = torch.transpose(fbank_delta_delta, 0, 1)
        fbanks = torch.cat([fbanks, fbank_delta, fbank_delta_delta], dim=1)

        fbanks =fbanks.unsqueeze(0).to(device)

        Out_List = model(fbanks)
        pred_mask = Out_List[-1]
        spectrum = stft(audio, args)
        pred_mask = pred_mask.permute(2,1,0) 
        masked_spec = spectrum * pred_mask.detach().cpu()
        masked_spec_complex = masked_spec[:,:,0] + 1j*masked_spec[:,:,1]
        outputs = istft(masked_spec_complex, args, len(audio))

    return outputs.numpy() / MAX_WAV_VALUE
