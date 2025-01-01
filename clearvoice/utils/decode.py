#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
# Authors: Shengkui Zhao, Zexu Pan

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

# Constant for normalizing audio values
MAX_WAV_VALUE = 32768.0

def decode_one_audio(model, device, inputs, args, extract_noise=False):
    """Select and call the appropriate decoding function based on network type.
    
    Args:
        model: The model to use for decoding
        device: The device to run inference on
        inputs: Input audio data
        args: Arguments containing model configuration
        extract_noise: Whether to extract noise signal
    """
    if args.network == 'FRCRN_SE_16K':
        return decode_one_audio_frcrn_se_16k(model, device, inputs, args, extract_noise)
    elif args.network == 'MossFormer2_SE_48K':
        return decode_one_audio_mossformer2_se_48k(model, device, inputs, args, extract_noise)
    elif args.network == 'MossFormerGAN_SE_16K':
        return decode_one_audio_mossformergan_se_16k(model, device, inputs, args, extract_noise)
    elif args.network == 'MossFormer2_SS_16K':
        return decode_one_audio_mossformer2_ss_16k(model, device, inputs, args, extract_noise)
    else:
        print("No network found!")
        return

def decode_one_audio_mossformer2_ss_16k(model, device, inputs, args, extract_noise=False):
    """Decodes audio using the MossFormer2 model for speech separation at 16kHz.

    This function handles the audio decoding process by processing the input tensor
    in segments, if necessary, and applies the model to obtain separated audio outputs.

    Args:
        model (nn.Module): The trained MossFormer2 model for decoding.
        device (torch.device): The device (CPU or GPU) to perform computations on.
        inputs (torch.Tensor): Input audio tensor of shape (B, T), where B is the batch size
                              and T is the number of time steps.
        args (Namespace): Contains arguments for decoding configuration.
        extract_noise (bool): Whether to extract noise signal

    Returns:
        list: A list of decoded audio outputs for each speaker.
    """
    out = []  # Initialize the list to store outputs
    decode_do_segment = False  # Flag to determine if segmentation is needed
    window = args.sampling_rate * args.decode_window  # Decoding window length
    stride = int(window * 0.75)  # Decoding stride if segmentation is used
    b, t = inputs.shape  # Get batch size and input length

    rms_input = (inputs ** 2).mean() ** 0.5

    # Check if input length exceeds one-time decode length to decide on segmentation
    if t > args.sampling_rate * args.one_time_decode_length:
        decode_do_segment = True  # Enable segment decoding for long sequences

    # Pad the inputs to ensure they meet the decoding window length requirements
    if t < window:
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], window - t))], axis=1)
    elif t < window + stride:
        padding = window + stride - t
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], axis=1)
    else:
        if (t - window) % stride != 0:
            padding = t - (t - window) // stride * stride
            inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], axis=1)

    inputs = torch.from_numpy(np.float32(inputs)).to(device)  # Convert inputs to torch tensor and move to device
    b, t = inputs.shape  # Update batch size and input length after conversion

    # Process the inputs in segments if necessary
    if decode_do_segment:
        outputs = np.zeros((args.num_spks, t))  # Initialize output array for each speaker
        give_up_length = (window - stride) // 2  # Calculate length to give up at each segment
        current_idx = 0  # Initialize current index for segmentation
        while current_idx + window <= t:
            tmp_input = inputs[:, current_idx:current_idx + window]  # Get segment input
            tmp_out_list = model(tmp_input)  # Forward pass through the model
            for spk in range(args.num_spks):
                # Convert output for the current speaker to numpy
                tmp_out_list[spk] = tmp_out_list[spk][0, :].detach().cpu().numpy()
                if current_idx == 0:
                    # For the first segment, use the whole segment minus the give-up length
                    outputs[spk, current_idx:current_idx + window - give_up_length] = tmp_out_list[spk][:-give_up_length]
                else:
                    # For subsequent segments, account for the give-up length at both ends
                    outputs[spk, current_idx + give_up_length:current_idx + window - give_up_length] = tmp_out_list[spk][give_up_length:-give_up_length]
            current_idx += stride  # Move to the next segment
        for spk in range(args.num_spks):
            out.append(outputs[spk, :])  # Append outputs for each speaker
    else:
        # If no segmentation is required, process the entire input
        out_list = model(inputs)
        for spk in range(args.num_spks):
            out.append(out_list[spk][0, :].detach().cpu().numpy())  # Append output for each speaker

    # Normalize the outputs back to the input magnitude for each speaker
    for spk in range(args.num_spks):
        rms_out = (out[spk] ** 2).mean() ** 0.5
        out[spk] = out[spk] / rms_out * rms_input
    return out  # Return the list of normalized outputs

def decode_one_audio_frcrn_se_16k(model, device, inputs, args, extract_noise=False):
    """Decodes audio using the FRCRN model for speech enhancement at 16kHz.

    Args:
        model: The trained FRCRN model used for decoding
        device: The device to perform computations on
        inputs: Input audio tensor of shape (B, T)
        args: Arguments containing model configuration
        extract_noise: Whether to extract noise signal

    Returns:
        If extract_noise is True:
            tuple: (enhanced_audio, noise_audio)
        else:
            ndarray: enhanced_audio
    """
    decode_do_segment = False
    window = args.sampling_rate * args.decode_window
    stride = int(window * 0.75)
    b, t = inputs.shape

    # 检查是否需要分段处理
    if t > args.sampling_rate * args.one_time_decode_length:
        decode_do_segment = True

    # 转换输入为PyTorch张量并保存原始输入
    original_inputs = torch.from_numpy(np.float32(inputs)).to(device)
    inputs = original_inputs.clone()

    if decode_do_segment:
        outputs = np.zeros(t)
        if extract_noise:
            noise_outputs = np.zeros(t)
        give_up_length = (window - stride) // 2
        current_idx = 0

        while current_idx + window <= t:
            tmp_input = inputs[:, current_idx:current_idx + window]
            # 获取增强后的语音
            tmp_output = model.inference(tmp_input).detach().cpu().numpy()

            if current_idx == 0:
                outputs[current_idx:current_idx + window - give_up_length] = tmp_output[:-give_up_length]
            else:
                outputs[current_idx + give_up_length:current_idx + window - give_up_length] = \
                    tmp_output[give_up_length:-give_up_length]

            current_idx += stride

        if extract_noise:
            # 计算噪音信号
            original = original_inputs.cpu().numpy()[0, :t]
            noise_outputs = original - outputs
            return outputs, noise_outputs
        return outputs
    else:
        # 处理完整音频
        enhanced = model.inference(inputs).detach().cpu().numpy()
        
        if extract_noise:
            # 计算噪音信号
            original = original_inputs.cpu().numpy()[0, :t]
            noise = original - enhanced
            return enhanced, noise
        return enhanced

def decode_one_audio_mossformergan_se_16k(model, device, inputs, args, extract_noise=False):
    """Decodes audio using the MossFormerGAN model for speech enhancement at 16kHz.

    This function processes the input audio tensor either in segments or as a whole,
    depending on the length of the input. The `_decode_one_audio_mossformergan_se_16k`
    function is called to perform the model inference and return the enhanced audio output.

    Args:
        model (nn.Module): The trained MossFormerGAN model used for decoding.
        device (torch.device): The device (CPU or GPU) for computation.
        inputs (torch.Tensor): Input audio tensor of shape (B, T).
        args (Namespace): Contains arguments for decoding configuration.
        extract_noise (bool): Whether to extract noise signal (default: False)

    Returns:
        If extract_noise is True:
            tuple: (enhanced_audio, noise_audio)
        else:
            ndarray: enhanced_audio
    """
    decode_do_segment = False
    window = args.sampling_rate * args.decode_window
    stride = int(window * 0.75)
    b, t = inputs.shape

    # Check if input length exceeds one-time decode length to decide on segmentation
    if t > args.sampling_rate * args.one_time_decode_length:
        decode_do_segment = True

    # Convert inputs to PyTorch tensor and compute normalization factor
    inputs = torch.from_numpy(np.float32(inputs)).to(device)
    norm_factor = torch.sqrt(inputs.size(-1) / torch.sum((inputs ** 2.0), dim=-1))
    b, t = inputs.shape

    if decode_do_segment:
        outputs = np.zeros(t)
        if extract_noise:
            noise_outputs = np.zeros(t)
        give_up_length = (window - stride) // 2
        current_idx = 0

        while current_idx + window <= t:
            tmp_input = inputs[:, current_idx:current_idx + window]
            if extract_noise:
                tmp_output, tmp_noise = _decode_one_audio_mossformergan_se_16k(
                    model, device, tmp_input, norm_factor, args, extract_noise)
            else:
                tmp_output = _decode_one_audio_mossformergan_se_16k(
                    model, device, tmp_input, norm_factor, args, extract_noise)

            if current_idx == 0:
                outputs[current_idx:current_idx + window - give_up_length] = tmp_output[:-give_up_length]
                if extract_noise:
                    noise_outputs[current_idx:current_idx + window - give_up_length] = tmp_noise[:-give_up_length]
            else:
                outputs[current_idx + give_up_length:current_idx + window - give_up_length] = \
                    tmp_output[give_up_length:-give_up_length]
                if extract_noise:
                    noise_outputs[current_idx + give_up_length:current_idx + window - give_up_length] = \
                        tmp_noise[give_up_length:-give_up_length]

            current_idx += stride

        if extract_noise:
            return outputs, noise_outputs
        return outputs
    else:
        if extract_noise:
            enhanced, noise = _decode_one_audio_mossformergan_se_16k(
                model, device, inputs, norm_factor, args, extract_noise)
            return enhanced, noise
        else:
            return _decode_one_audio_mossformergan_se_16k(
                model, device, inputs, norm_factor, args, extract_noise)

@torch.no_grad()
def _decode_one_audio_mossformergan_se_16k(model, device, inputs, norm_factor, args, extract_noise=False):
    """Processes audio inputs through the MossFormerGAN model for speech enhancement.

    This function performs the following steps:
    1. Pads the input audio tensor to fit the model requirements.
    2. Computes a normalization factor for the input tensor.
    3. Applies Short-Time Fourier Transform (STFT) to convert the audio into the frequency domain.
    4. Processes the STFT representation through the model to predict the real and imaginary components.
    5. Uncompresses the predicted spectrogram and applies Inverse STFT (iSTFT) to convert back to time domain audio.
    6. Normalizes the output audio.

    Args:
        model (nn.Module): The trained MossFormerGAN model used for decoding.
        device (torch.device): The device (CPU or GPU) for computation.
        inputs (torch.Tensor): Input audio tensor of shape (B, T).
        norm_factor (torch.Tensor): A norm tensor to regularize input amplitude.
        args (Namespace): Contains arguments for STFT parameters and normalization.
        extract_noise (bool): Whether to extract noise signal (default: False)

    Returns:
        If extract_noise is True:
            tuple: (enhanced_audio, noise_audio)
        else:
            ndarray: enhanced_audio
    """
    input_len = inputs.size(-1)
    nframe = int(np.ceil(input_len / args.win_inc))
    padded_len = nframe * args.win_inc
    padding_len = padded_len - input_len

    # Save original input for noise calculation if needed
    original_inputs = inputs.clone()

    # Pad inputs
    inputs = torch.cat([inputs, inputs[:, :padding_len]], dim=-1)
    inputs = torch.transpose(inputs, 0, 1)
    inputs = torch.transpose(inputs * norm_factor, 0, 1)

    # Compute STFT
    inputs_spec = stft(inputs, args, center=True, periodic=True, onesided=True)
    inputs_spec = inputs_spec.to(torch.float32)

    # Compress power spectrum
    inputs_spec = power_compress(inputs_spec).permute(0, 1, 3, 2)

    # Get model predictions
    out_list = model(inputs_spec)
    pred_real, pred_imag = out_list[0].permute(0, 1, 3, 2), out_list[1].permute(0, 1, 3, 2)

    # Uncompress predicted spectrum
    pred_spec_uncompress = power_uncompress(pred_real, pred_imag).squeeze(1)

    # Reconstruct enhanced audio
    outputs = istft(pred_spec_uncompress, args, center=True, periodic=True, onesided=True)
    outputs = outputs.squeeze(0) / norm_factor
    enhanced = outputs[:input_len]

    if extract_noise:
        # Calculate noise signal
        original = original_inputs.squeeze(0)[:input_len]
        noise = original - enhanced
        return enhanced.detach().cpu().numpy(), noise.detach().cpu().numpy()

    return enhanced.detach().cpu().numpy()

def decode_one_audio_mossformer2_se_48k(model, device, inputs, args, extract_noise=False):
    """Processes audio inputs through the MossFormer2 model for speech enhancement at 48kHz.

    This function decodes audio input using the following steps:
    1. Normalizes the audio input to a maximum WAV value.
    2. Checks the length of the input to decide between online decoding and batch processing.
    3. For longer inputs, processes the audio in segments using a sliding window.
    4. Computes filter banks and their deltas for the audio segment.
    5. Passes the filter banks through the model to get a predicted mask.
    6. Applies the mask to the spectrogram of the audio segment and reconstructs the audio.
    7. For shorter inputs, processes them in one go without segmentation.

    Args:
        model (nn.Module): The trained MossFormer2 model used for decoding.
        device (torch.device): The device (CPU or GPU) for computation.
        inputs (torch.Tensor): Input audio tensor of shape (B, T).
        args (Namespace): Contains arguments for sampling rate, window size, and other parameters.
        extract_noise (bool): Whether to extract noise signal (default: False)

    Returns:
        If extract_noise is True:
            tuple: (enhanced_audio, noise_audio)
        else:
            ndarray: enhanced_audio normalized to [-1, 1]
    """
    inputs = inputs[0, :]
    input_len = inputs.shape[0]
    inputs = inputs * MAX_WAV_VALUE

    if input_len > args.sampling_rate * args.one_time_decode_length:  # 20 seconds
        online_decoding = True
        if online_decoding:
            window = int(args.sampling_rate * args.decode_window)  # Define window length
            stride = int(window * 0.75)  # Define stride length
            t = inputs.shape[0]

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

            audio = torch.from_numpy(inputs).type(torch.FloatTensor)
            t = audio.shape[0]
            outputs = torch.from_numpy(np.zeros(t))
            if extract_noise:
                noise_outputs = torch.from_numpy(np.zeros(t))
            
            give_up_length = (window - stride) // 2
            dfsmn_memory_length = 0
            current_idx = 0

            while current_idx + window <= t:
                # Select appropriate segment of audio for processing
                if current_idx < dfsmn_memory_length:
                    audio_segment = audio[0:current_idx + window]
                else:
                    audio_segment = audio[current_idx - dfsmn_memory_length:current_idx + window]

                # Compute filter banks and their deltas
                fbanks = compute_fbank(audio_segment.unsqueeze(0), args)
                fbank_tr = torch.transpose(fbanks, 0, 1)
                fbank_delta = torchaudio.functional.compute_deltas(fbank_tr)
                fbank_delta_delta = torchaudio.functional.compute_deltas(fbank_delta)
                fbank_delta = torch.transpose(fbank_delta, 0, 1)
                fbank_delta_delta = torch.transpose(fbank_delta_delta, 0, 1)
                
                fbanks = torch.cat([fbanks, fbank_delta, fbank_delta_delta], dim=1)
                fbanks = fbanks.unsqueeze(0).to(device)

                # Model inference
                Out_List = model(fbanks)
                pred_mask = Out_List[-1]
                spectrum = stft(audio_segment, args)
                pred_mask = pred_mask.permute(2, 1, 0)

                # Process enhanced audio
                masked_spec = spectrum.cpu() * pred_mask.detach().cpu()
                masked_spec_complex = masked_spec[:, :, 0] + 1j * masked_spec[:, :, 1]
                output_segment = istft(masked_spec_complex, args, len(audio_segment))

                # Process noise if requested
                if extract_noise:
                    noise_mask = 1 - pred_mask.detach().cpu()
                    noise_spec = spectrum.cpu() * noise_mask
                    noise_spec_complex = noise_spec[:, :, 0] + 1j * noise_spec[:, :, 1]
                    noise_segment = istft(noise_spec_complex, args, len(audio_segment))

                # Store results
                if current_idx == 0:
                    outputs[current_idx:current_idx + window - give_up_length] = output_segment[:-give_up_length]
                    if extract_noise:
                        noise_outputs[current_idx:current_idx + window - give_up_length] = noise_segment[:-give_up_length]
                else:
                    output_segment = output_segment[-window:]
                    outputs[current_idx + give_up_length:current_idx + window - give_up_length] = \
                        output_segment[give_up_length:-give_up_length]
                    if extract_noise:
                        noise_segment = noise_segment[-window:]
                        noise_outputs[current_idx + give_up_length:current_idx + window - give_up_length] = \
                            noise_segment[give_up_length:-give_up_length]

                current_idx += stride

            if extract_noise:
                return outputs.numpy() / MAX_WAV_VALUE, noise_outputs.numpy() / MAX_WAV_VALUE
            return outputs.numpy() / MAX_WAV_VALUE

    else:
        # Process shorter audio in one go
        audio = torch.from_numpy(inputs).type(torch.FloatTensor)
        
        # Compute filter banks and their deltas
        fbanks = compute_fbank(audio.unsqueeze(0), args)
        fbank_tr = torch.transpose(fbanks, 0, 1)
        fbank_delta = torchaudio.functional.compute_deltas(fbank_tr)
        fbank_delta_delta = torchaudio.functional.compute_deltas(fbank_delta)
        fbank_delta = torch.transpose(fbank_delta, 0, 1)
        fbank_delta_delta = torch.transpose(fbank_delta_delta, 0, 1)
        
        fbanks = torch.cat([fbanks, fbank_delta, fbank_delta_delta], dim=1)
        fbanks = fbanks.unsqueeze(0).to(device)

        # Model inference
        Out_List = model(fbanks)
        pred_mask = Out_List[-1]
        spectrum = stft(audio, args)
        pred_mask = pred_mask.permute(2, 1, 0)

        # Process enhanced audio
        masked_spec = spectrum * pred_mask.detach().cpu()
        masked_spec_complex = masked_spec[:, :, 0] + 1j * masked_spec[:, :, 1]
        enhanced = istft(masked_spec_complex, args, len(audio))

        if extract_noise:
            # Calculate noise signal
            noise_mask = 1 - pred_mask.detach().cpu()
            noise_spec = spectrum * noise_mask
            noise_spec_complex = noise_spec[:, :, 0] + 1j * noise_spec[:, :, 1]
            noise = istft(noise_spec_complex, args, len(audio))
            
            return enhanced.numpy() / MAX_WAV_VALUE, noise.numpy() / MAX_WAV_VALUE

        return enhanced.numpy() / MAX_WAV_VALUE

def decode_one_audio_AV_MossFormer2_TSE_16K(model, inputs, args):
    """Processes video inputs through the AV mossformer2 model with Target speaker extraction (TSE) for decoding at 16kHz.

    This function decodes audio input using the following steps:
    1. Checks if the input audio length requires segmentation or can be processed in one go.
    2. If the input audio is long enough, processes it in overlapping segments using a sliding window approach.
    3. Applies the model to each segment or the entire input, and collects the output.

    Args:
        model (nn.Module): The trained SpEx model for speech enhancement.
        inputs (numpy.ndarray): Input audio and visual data
        args (Namespace): Contains arguments for sampling rate, window size, and other parameters.

    Returns:
        numpy.ndarray: The decoded audio output as a NumPy array.
    """

    audio, visual = inputs
    max_val = np.max(np.abs(audio))
    if max_val > 1:
        audio /= max_val
    
    b, t = audio.shape  # Get batch size (b) and input length (t)

    decode_do_segement = False  # Flag to determine if segmentation is needed
    # Check if the input length exceeds the defined threshold for segmentation
    if t > args.sampling_rate * args.one_time_decode_length:
        decode_do_segement = True  # Enable segmentation for long inputs

    # Convert inputs to a PyTorch tensor and move to the specified device
    audio = torch.from_numpy(np.float32(audio)).to(args.device)
    visual = torch.from_numpy(np.float32(visual)).to(args.device)

    print(audio.shape)
    print(visual.shape)

    if decode_do_segement:
        print('********')
        outputs = np.zeros(t)  # Initialize output array
        window = args.sampling_rate * args.decode_window  # Window length for processing
        window_v = 25 * args.decode_window
        stride = int(window * 0.6)  # Decoding stride for segmenting the input
        give_up_length = (window - stride) // 2  # Calculate length to give up at each segment
        current_idx = 0  # Initialize current index for sliding window

        # Process the audio in overlapping segments
        while current_idx + window < t:
            tmp_audio = audio[:, current_idx:current_idx + window]  # Select current audio segment

            current_idx_v = int(current_idx/args.sampling_rate*25)  # Select current video segment index
            tmp_video = visual[:, current_idx_v:current_idx_v + window_v, :, :] # Select current video segment
            
            tmp_output = model(tmp_audio, tmp_video).detach().squeeze().cpu().numpy()  # Apply model to the segment
            
            # For the first segment, use the whole segment minus the give-up length
            if current_idx == 0:
                outputs[current_idx:current_idx + window - give_up_length] = tmp_output[:-give_up_length]
            else:
                # For subsequent segments, account for the give-up length
                outputs[current_idx + give_up_length:current_idx + window - give_up_length] = tmp_output[give_up_length:-give_up_length]

            current_idx += stride  # Move to the next segment

        # Process the last window of audio
        tmp_audio = audio[:, -window:]
        tmp_video = visual[:, -window_v:, :, :]
        tmp_output = model(tmp_audio, tmp_video).detach().squeeze().cpu().numpy()  # Apply model to the segment
        outputs[-window + give_up_length:] = tmp_output[give_up_length:]
    else:
        # Process the entire input at once if segmentation is not needed
        outputs = model(audio, visual).detach().squeeze().cpu().numpy()


    return outputs  # Return the decoded audio output as a NumPy array
