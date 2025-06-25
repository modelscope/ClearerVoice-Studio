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
from .misc import power_compress, power_uncompress, stft, istft, compute_fbank
from .bandwidth_sub import bandwidth_sub
from ..dataloader.meldataset import mel_spectrogram

# Constant for normalizing audio values
MAX_WAV_VALUE = 32768.0

def decode_one_audio(model, device, inputs, args):
    """Decodes audio using the specified model based on the provided network type.

    This function selects the appropriate decoding function based on the specified
    network in the arguments and processes the input audio data accordingly.

    Args:
        model (nn.Module): The trained model used for decoding.
        device (torch.device): The device (CPU or GPU) to perform computations on.
        inputs (torch.Tensor): Input audio tensor.
        args (Namespace): Contains arguments for network configuration.

    Returns:
        list: A list of decoded audio outputs for each speaker.
    """
    # Select decoding function based on the network type specified in args
    if args.network == 'FRCRN_SE_16K':
        return decode_one_audio_frcrn_se_16k(model, device, inputs, args)
    elif args.network == 'MossFormer2_SE_48K':
        return decode_one_audio_mossformer2_se_48k(model, device, inputs, args)
    elif args.network == 'MossFormerGAN_SE_16K':
        return decode_one_audio_mossformergan_se_16k(model, device, inputs, args)
    elif args.network == 'MossFormer2_SS_16K':
        return decode_one_audio_mossformer2_ss_16k(model, device, inputs, args)
    elif args.network == 'MossFormer2_SR_48K':
        return decode_one_audio_mossformer2_sr_48k(model, device, inputs, args)
    else:
        print("No network found!")  # Print error message if no valid network is specified
        return 

def decode_one_audio_mossformer2_ss_16k(model, device, inputs, args):
    """Decodes audio using the MossFormer2 model for speech separation at 16kHz.

    This function handles the audio decoding process by processing the input tensor
    in segments, if necessary, and applies the model to obtain separated audio outputs.

    Args:
        model (nn.Module): The trained MossFormer2 model for decoding.
        device (torch.device): The device (CPU or GPU) to perform computations on.
        inputs (torch.Tensor): Input audio tensor of shape (B, T), where B is the batch size
                              and T is the number of time steps.
        args (Namespace): Contains arguments for decoding configuration.

    Returns:
        list: A list of decoded audio outputs for each speaker.
    """
    out = []  # Initialize the list to store outputs
    decode_do_segment = False  # Flag to determine if segmentation is needed
    window = int(args.sampling_rate * args.decode_window)  # Decoding window length
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

def decode_one_audio_frcrn_se_16k(model, device, inputs, args):
    """Decodes audio using the FRCRN model for speech enhancement at 16kHz.

    This function processes the input audio tensor either in segments or as a whole, 
    depending on the length of the input. The model's inference method is applied 
    to obtain the enhanced audio output.

    Args:
        model (nn.Module): The trained FRCRN model used for decoding.
        device (torch.device): The device (CPU or GPU) to perform computations on.
        inputs (torch.Tensor): Input audio tensor of shape (B, T), where B is the batch size
                              and T is the number of time steps.
        args (Namespace): Contains arguments for decoding configuration.

    Returns:
        numpy.ndarray: The decoded audio output, which has been enhanced by the model.
    """
    decode_do_segment = False  # Flag to determine if segmentation is needed

    window = int(args.sampling_rate * args.decode_window)  # Decoding window length
    stride = int(window * 0.75)  # Decoding stride for segmenting the input
    b, t = inputs.shape  # Get batch size (b) and input length (t)

    # Check if input length exceeds one-time decode length to decide on segmentation
    if t > args.sampling_rate * args.one_time_decode_length:
        decode_do_segment = True  # Enable segment decoding for long sequences

    # Pad the inputs to meet the decoding window length requirements
    if t < window:
        # Pad with zeros if the input length is less than the window size
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], window - t))], axis=1)
    elif t < window + stride:
        # Pad the input if its length is less than the window plus stride
        padding = window + stride - t
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], axis=1)
    else:
        # Ensure the input length is a multiple of the stride
        if (t - window) % stride != 0:
            padding = t - (t - window) // stride * stride
            inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], axis=1)

    # Convert inputs to a PyTorch tensor and move to the specified device
    inputs = torch.from_numpy(np.float32(inputs)).to(device)
    b, t = inputs.shape  # Update batch size and input length after conversion

    # Process the inputs in segments if necessary
    if decode_do_segment:
        outputs = np.zeros(t)  # Initialize the output array
        give_up_length = (window - stride) // 2  # Calculate length to give up at each segment
        current_idx = 0  # Initialize current index for segmentation

        while current_idx + window <= t:
            tmp_input = inputs[:, current_idx:current_idx + window]  # Get segment input
            tmp_output = model.inference(tmp_input).detach().cpu().numpy()  # Inference on segment

            # For the first segment, use the whole segment minus the give-up length
            if current_idx == 0:
                outputs[current_idx:current_idx + window - give_up_length] = tmp_output[:-give_up_length]
            else:
                # For subsequent segments, account for the give-up length
                outputs[current_idx + give_up_length:current_idx + window - give_up_length] = tmp_output[give_up_length:-give_up_length]

            current_idx += stride  # Move to the next segment
    else:
        # If no segmentation is required, process the entire input
        outputs = model.inference(inputs).detach().cpu().numpy()  # Inference on full input

    return outputs  # Return the decoded audio output

def decode_one_audio_mossformergan_se_16k(model, device, inputs, args):
    """Decodes audio using the MossFormerGAN model for speech enhancement at 16kHz.

    This function processes the input audio tensor either in segments or as a whole, 
    depending on the length of the input. The `_decode_one_audio_mossformergan_se_16k` 
    function is called to perform the model inference and return the enhanced audio output.

    Args:
        model (nn.Module): The trained MossFormerGAN model used for decoding.
        device (torch.device): The device (CPU or GPU) for computation.
        inputs (torch.Tensor): Input audio tensor of shape (B, T), where B is the batch size 
                              and T is the number of time steps.
        args (Namespace): Contains arguments for decoding configuration.

    Returns:
        numpy.ndarray: The decoded audio output, which has been enhanced by the model.
    """
    decode_do_segment = False  # Flag to determine if segmentation is needed
    window = int(args.sampling_rate * args.decode_window)  # Decoding window length
    stride = int(window * 0.75)  # Decoding stride for segmenting the input
    b, t = inputs.shape  # Get batch size (b) and input length (t)

    # Check if input length exceeds one-time decode length to decide on segmentation
    if t > args.sampling_rate * args.one_time_decode_length:
        decode_do_segment = True  # Enable segment decoding for long sequences

    # Convert inputs to a PyTorch tensor and move to the specified device
    inputs = torch.from_numpy(np.float32(inputs)).to(device)

    # Compute normalization factor based on the input
    norm_factor = torch.sqrt(inputs.size(-1) / torch.sum((inputs ** 2.0), dim=-1))

    b, t = inputs.shape  # Update batch size and input length after conversion

    # Process the inputs in segments if necessary
    if decode_do_segment:
        outputs = np.zeros(t)  # Initialize the output array
        give_up_length = (window - stride) // 2  # Calculate length to give up at each segment
        current_idx = 0  # Initialize current index for segmentation

        while current_idx + window <= t:
            tmp_input = inputs[:, current_idx:current_idx + window]  # Get segment input
            tmp_output = _decode_one_audio_mossformergan_se_16k(model, device, tmp_input, norm_factor, args)  # Inference on segment

            # For the first segment, use the whole segment minus the give-up length
            if current_idx == 0:
                outputs[current_idx:current_idx + window - give_up_length] = tmp_output[:-give_up_length]
            else:
                # For subsequent segments, account for the give-up length
                outputs[current_idx + give_up_length:current_idx + window - give_up_length] = tmp_output[give_up_length:-give_up_length]

            current_idx += stride  # Move to the next segment
        # Handle the remaining part of the input if it doesn't fit into a full segment
        # current_idx > t - window
        if current_idx < t:
            last_start = current_idx - give_up_length # shift left by give_up_length
            tmp_input = inputs[:, last_start:]
            tmp_output = _decode_one_audio_mossformergan_se_16k(model, device, tmp_input, norm_factor, args)
            outputs[current_idx:] = tmp_output[give_up_length:]  # Fill the remaining part of the output
        return outputs  # Return the accumulated outputs from segments
    else:
        # If no segmentation is required, process the entire input
        return _decode_one_audio_mossformergan_se_16k(model, device, inputs, norm_factor, args)  # Inference on full input

@torch.no_grad()
def _decode_one_audio_mossformergan_se_16k(model, device, inputs, norm_factor, args):
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
        inputs (torch.Tensor): Input audio tensor of shape (B, T), where B is the batch size and T is the number of time steps.
        norm_factor (torch.Tensor): A norm tensor to regularize input amplitude
        args (Namespace): Contains arguments for STFT parameters and normalization.

    Returns:
        numpy.ndarray: The decoded audio output, which has been enhanced by the model.
    """
    input_len = inputs.size(-1)  # Get the length of the input audio
    nframe = int(np.ceil(input_len / args.win_inc))  # Calculate the number of frames based on window increment
    padded_len = int(nframe * args.win_inc)  # Calculate the padded length to fit the model
    padding_len = padded_len - input_len  # Determine how much padding is needed

    # Pad the input audio with the beginning of the input
    inputs = torch.cat([inputs, inputs[:, :padding_len]], dim=-1)

    # Prepare inputs for STFT by transposing and normalizing
    inputs = torch.transpose(inputs, 0, 1)  # Change shape for STFT
    inputs = torch.transpose(inputs * norm_factor, 0, 1)  # Apply normalization factor and transpose back

    # Perform Short-Time Fourier Transform (STFT) on the normalized inputs
    inputs_spec = stft(inputs, args, center=True, periodic=True, onesided=True)
    inputs_spec = inputs_spec.to(torch.float32)  # Ensure the spectrogram is in float32 format

    # Compress the power of the spectrogram to improve model performance
    inputs_spec = power_compress(inputs_spec).permute(0, 1, 3, 2)

    # Pass the compressed spectrogram through the model to get predicted real and imaginary parts
    out_list = model(inputs_spec)
    pred_real, pred_imag = out_list[0].permute(0, 1, 3, 2), out_list[1].permute(0, 1, 3, 2)

    # Uncompress the predicted spectrogram to get the magnitude and phase
    pred_spec_uncompress = power_uncompress(pred_real, pred_imag).squeeze(1)

    # Perform Inverse STFT (iSTFT) to convert back to time domain audio
    outputs = istft(pred_spec_uncompress, args, center=True, periodic=True, onesided=True)

    # Normalize the output audio by dividing by the normalization factor
    outputs = outputs.squeeze(0) / norm_factor

    return outputs[:input_len].detach().cpu().numpy()  # Return the output as a numpy array

def decode_one_audio_mossformer2_se_48k(model, device, inputs, args):
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
        inputs (torch.Tensor): Input audio tensor of shape (B, T), where B is the batch size and T is the number of time steps.
        args (Namespace): Contains arguments for sampling rate, window size, and other parameters.

    Returns:
        numpy.ndarray: The decoded audio output, normalized to the range [-1, 1].
    """
    inputs = inputs[0, :]  # Extract the first element from the input tensor
    input_len = inputs.shape[0]  # Get the length of the input audio
    inputs = inputs * MAX_WAV_VALUE  # Normalize the input to the maximum WAV value

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

                # Compute filter banks for the audio segment
                fbanks = compute_fbank(audio_segment.unsqueeze(0), args)
                
                # Compute deltas for filter banks
                fbank_tr = torch.transpose(fbanks, 0, 1)  # Transpose for delta computation
                fbank_delta = torchaudio.functional.compute_deltas(fbank_tr)  # First-order delta
                fbank_delta_delta = torchaudio.functional.compute_deltas(fbank_delta)  # Second-order delta
                
                # Transpose back to original shape
                fbank_delta = torch.transpose(fbank_delta, 0, 1)
                fbank_delta_delta = torch.transpose(fbank_delta_delta, 0, 1)

                # Concatenate the original filter banks with their deltas
                fbanks = torch.cat([fbanks, fbank_delta, fbank_delta_delta], dim=1)
                fbanks = fbanks.unsqueeze(0).to(device)  # Add batch dimension and move to device

                # Pass filter banks through the model
                Out_List = model(fbanks)
                pred_mask = Out_List[-1]  # Get the predicted mask from the output

                # Apply STFT to the audio segment
                spectrum = stft(audio_segment, args)
                pred_mask = pred_mask.permute(2, 1, 0)  # Permute dimensions for masking
                masked_spec = spectrum.cpu() * pred_mask.detach().cpu()  # Apply mask to the spectrum
                masked_spec_complex = masked_spec[:, :, 0] + 1j * masked_spec[:, :, 1]  # Convert to complex form

                # Reconstruct audio from the masked spectrogram
                output_segment = istft(masked_spec_complex, args, len(audio_segment))

                # Store the output segment in the output tensor
                if current_idx == 0:
                    outputs[current_idx:current_idx + window - give_up_length] = output_segment[:-give_up_length]
                else:
                    output_segment = output_segment[-window:]  # Get the latest window of output
                    outputs[current_idx + give_up_length:current_idx + window - give_up_length] = output_segment[give_up_length:-give_up_length]
                
                current_idx += stride  # Move to the next segment

    else:
        # Process the entire audio at once if it is shorter than the threshold
        audio = torch.from_numpy(inputs).type(torch.FloatTensor)
        fbanks = compute_fbank(audio.unsqueeze(0), args)

        # Compute deltas for filter banks
        fbank_tr = torch.transpose(fbanks, 0, 1)
        fbank_delta = torchaudio.functional.compute_deltas(fbank_tr)
        fbank_delta_delta = torchaudio.functional.compute_deltas(fbank_delta)
        fbank_delta = torch.transpose(fbank_delta, 0, 1)
        fbank_delta_delta = torch.transpose(fbank_delta_delta, 0, 1)

        # Concatenate the original filter banks with their deltas
        fbanks = torch.cat([fbanks, fbank_delta, fbank_delta_delta], dim=1)
        fbanks = fbanks.unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Pass filter banks through the model
        Out_List = model(fbanks)
        pred_mask = Out_List[-1]  # Get the predicted mask
        spectrum = stft(audio, args)  # Apply STFT to the audio
        pred_mask = pred_mask.permute(2, 1, 0)  # Permute dimensions for masking
        masked_spec = spectrum * pred_mask.detach().cpu()  # Apply mask to the spectrum
        masked_spec_complex = masked_spec[:, :, 0] + 1j * masked_spec[:, :, 1]  # Convert to complex form
        
        # Reconstruct audio from the masked spectrogram
        outputs = istft(masked_spec_complex, args, len(audio))

    return outputs.numpy() / MAX_WAV_VALUE  # Return the output normalized to [-1, 1]

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
