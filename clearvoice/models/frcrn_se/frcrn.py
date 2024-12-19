import torch.nn as nn
import torch 
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.dirname(__file__))
from models.frcrn_se.conv_stft import ConvSTFT, ConviSTFT
import numpy as np
from models.frcrn_se.unet import UNet

class FRCRN_Wrapper_StandAlone(nn.Module):
    """
    A wrapper class for the DCCRN model used in standalone mode.

    This class initializes the DCCRN model with predefined parameters and provides a forward method to process
    input audio signals for speech enhancement.

    Args:
        args: Arguments containing model configuration (not used in this wrapper).
    """
    def __init__(self, args):
        super(FRCRN_Wrapper_StandAlone, self).__init__()
        # Initialize the DCCRN model with specific parameters
        self.model = DCCRN(
            complex=True,
            model_complexity=45,
            model_depth=14,
            log_amp=False,
            padding_mode="zeros",
            win_len=640,
            win_inc=320,
            fft_len=640,
            win_type='hanning'
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor representing audio signals.

        Returns:
            torch.Tensor: Processed output tensor after applying the model.
        """
        output = self.model(x)       
        return output[1][0]  # Return estimated waveform


class FRCRN_SE_16K(nn.Module):
    """
    A class for the FRCRN model specifically configured for 16 kHz input signals.

    This class allows for customization of model parameters based on provided arguments.

    Args:
        args: Configuration parameters for the model.
    """
    def __init__(self, args):
        super(FRCRN_SE_16K, self).__init__()
        # Initialize the DCCRN model with parameters from args
        self.model = DCCRN(
            complex=True,
            model_complexity=45,
            model_depth=14,
            log_amp=False,
            padding_mode="zeros",
            win_len=args.win_len,
            win_inc=args.win_inc,
            fft_len=args.fft_len,
            win_type=args.win_type
        )

    def forward(self, x, real_time=False):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor representing audio signals.
            real_time (bool): Flag to indicate real-time processing.

        Returns:
            torch.Tensor: Processed output tensor after applying the model.
        """
        if real_time:
            return self.real_time_process(x)
        else:
            output = self.model(x)
            return output[1][0]  # Return estimated waveform

    def real_time_process(self, x):
        """
        Real-time processing method for the FRCRN model.

        Args:
            x (torch.Tensor): Input tensor representing audio signals.

        Returns:
            torch.Tensor: Processed output tensor after applying the model in real-time.
        """
        output = self.model.real_time_process(x)
        return output


class DCCRN(nn.Module):
    """
    We implemented our FRCRN model on the basis of DCCRN rep (https://github.com/huyanxin/DeepComplexCRN) for complex speech enhancement.

    The DCCRN model (Paper: https://arxiv.org/abs/2008.00264) employs a convolutional short-time Fourier transform (STFT) 
    and a UNet architecture for estimating clean speech from noisy inputs, FRCRN uses an enhanced
    Unet architecture.

    Args:
        complex (bool): Flag to determine whether to use complex numbers.
        model_complexity (int): Complexity level for the model.
        model_depth (int): Depth of the UNet model (14 or 20).
        log_amp (bool): Whether to use log amplitude to estimate signals.
        padding_mode (str): Padding mode for convolutions ('zeros', 'reflect').
        win_len (int): Window length for STFT.
        win_inc (int): Window increment for STFT.
        fft_len (int): FFT length.
        win_type (str): Window type for STFT (e.g., 'hanning').
    """
    def __init__(self, complex, model_complexity, model_depth, log_amp, padding_mode, win_len=400, win_inc=100, fft_len=512, win_type='hanning'):
        super().__init__()
        self.feat_dim = fft_len // 2 + 1

        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type

        # Initialize STFT and iSTFT layers
        fix = True  # Fixed STFT parameters
        self.stft = ConvSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, feature_type='complex', fix=fix)
        self.istft = ConviSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, feature_type='complex', fix=fix)

        # Initialize two UNet models for estimating complex masks
        self.unet = UNet(1, complex=complex, model_complexity=model_complexity, model_depth=model_depth, padding_mode=padding_mode)
        self.unet2 = UNet(1, complex=complex, model_complexity=model_complexity, model_depth=model_depth, padding_mode=padding_mode)

    def forward(self, inputs):
        """
        Forward pass of the FRCRN model.

        Args:
            inputs (torch.Tensor): Input tensor representing audio signals.

        Returns:
            list: A list containing estimated spectral features, waveform, and masks.
        """
        out_list = []
        # Compute the complex spectrogram using STFT
        cmp_spec = self.stft(inputs)  # [B, D*2, T]
        cmp_spec = torch.unsqueeze(cmp_spec, 1)  # [B, 1, D*2, T]
        
        # Split into real and imaginary parts
        cmp_spec = torch.cat([
            cmp_spec[:, :, :self.feat_dim, :],  # Real part
            cmp_spec[:, :, self.feat_dim:, :],  # Imaginary part
        ], 1)  # [B, 2, D, T]

        cmp_spec = torch.unsqueeze(cmp_spec, 4)  # [B, 2, D, T, 1]
        cmp_spec = torch.transpose(cmp_spec, 1, 4)  # [B, 1, D, T, 2]
        
        # Pass through the UNet to estimate masks
        unet1_out = self.unet(cmp_spec)  # First UNet output
        cmp_mask1 = torch.tanh(unet1_out)  # First mask

        unet2_out = self.unet2(unet1_out)  # Second UNet output
        cmp_mask2 = torch.tanh(unet2_out)  # Second mask
        cmp_mask2 = cmp_mask2 + cmp_mask1  # Combine masks

        # Apply the estimated mask to the complex spectrogram
        est_spec, est_wav, est_mask = self.apply_mask(cmp_spec, cmp_mask2)
        out_list.append(est_spec)
        out_list.append(est_wav)
        out_list.append(est_mask)
        return out_list

    def inference(self, inputs):
        """
        Inference method for the FRCRN model.

        This method performs a forward pass through the model to estimate the clean waveform
        from the noisy input.

        Args:
            inputs (torch.Tensor): Input tensor representing audio signals.

        Returns:
            torch.Tensor: Estimated waveform after processing.
        """
        # Compute the complex spectrogram using STFT
        cmp_spec = self.stft(inputs)  # [B, D*2, T]
        cmp_spec = torch.unsqueeze(cmp_spec, 1)  # [B, 1, D*2, T]

        # Split into real and imaginary parts
        cmp_spec = torch.cat([
            cmp_spec[:, :, :self.feat_dim, :],  # Real part
            cmp_spec[:, :, self.feat_dim:, :],  # Imaginary part
        ], 1)  # [B, 2, D, T]

        cmp_spec = torch.unsqueeze(cmp_spec, 4)  # [B, 2, D, T, 1]
        cmp_spec = torch.transpose(cmp_spec, 1, 4)  # [B, 1, D, T, 2]

        # Pass through the UNet to estimate masks
        unet1_out = self.unet(cmp_spec)
        cmp_mask1 = torch.tanh(unet1_out)
        
        unet2_out = self.unet2(unet1_out)
        cmp_mask2 = torch.tanh(unet2_out)
        cmp_mask2 = cmp_mask2 + cmp_mask1  # Combine masks

        # Apply the estimated mask to compute the estimated waveform
        _, est_wav, _ = self.apply_mask(cmp_spec, cmp_mask2)
        return est_wav[0]  # Return the estimated waveform

    def apply_mask(self, cmp_spec, cmp_mask):
        """
        Apply the estimated masks to the complex spectrogram.

        Args:
            cmp_spec (torch.Tensor): Complex spectrogram tensor.
            cmp_mask (torch.Tensor): Estimated mask tensor.

        Returns:
            tuple: Estimated spectrogram, waveform, and mask.
        """
        # Compute the estimated complex spectrogram using masks
        est_spec = torch.cat([
            cmp_spec[:, :, :, :, 0] * cmp_mask[:, :, :, :, 0] - cmp_spec[:, :, :, :, 1] * cmp_mask[:, :, :, :, 1],
            cmp_spec[:, :, :, :, 0] * cmp_mask[:, :, :, :, 1] + cmp_spec[:, :, :, :, 1] * cmp_mask[:, :, :, :, 0]
        ], 1)  # Combine real and imaginary parts
        
        est_spec = torch.cat([est_spec[:, 0, :, :], est_spec[:, 1, :, :]], 1)  # Flatten dimensions
        cmp_mask = torch.squeeze(cmp_mask, 1)
        cmp_mask = torch.cat([cmp_mask[:, :, :, 0], cmp_mask[:, :, :, 1]], 1)  # Combine masks

        est_wav = self.istft(est_spec)  # Inverse STFT to obtain waveform
        est_wav = torch.squeeze(est_wav, 1)  # Remove unnecessary dimensions
        return est_spec, est_wav, cmp_mask

    def get_params(self, weight_decay=0.0):
        """
        Get parameters for optimization with optional weight decay.

        Args:
            weight_decay (float): Weight decay for L2 regularization.

        Returns:
            list: List of dictionaries containing parameters and their weight decay settings.
        """
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def real_time_process(self, inputs):
        """
        Real-time processing method for the DCCRN model.

        Args:
            inputs (torch.Tensor): Input tensor representing audio signals.

        Returns:
            torch.Tensor: Processed output tensor after applying the model in real-time.
        """
        out_list = []
        # Compute the complex spectrogram using STFT
        cmp_spec = self.stft(inputs)  # [B, D*2, T]
        cmp_spec = torch.unsqueeze(cmp_spec, 1)  # [B, 1, D*2, T]
        
        # Split into real and imaginary parts
        cmp_spec = torch.cat([
            cmp_spec[:, :, :self.feat_dim, :],  # Real part
            cmp_spec[:, :, self.feat_dim:, :],  # Imaginary part
        ], 1)  # [B, 2, D, T]

        cmp_spec = torch.unsqueeze(cmp_spec, 4)  # [B, 2, D, T, 1]
        cmp_spec = torch.transpose(cmp_spec, 1, 4)  # [B, 1, D, T, 2]
        
        # Pass through the UNet to estimate masks
        unet1_out = self.unet(cmp_spec)  # First UNet output
        cmp_mask1 = torch.tanh(unet1_out)  # First mask

        unet2_out = self.unet2(unet1_out)  # Second UNet output
        cmp_mask2 = torch.tanh(unet2_out)  # Second mask
        cmp_mask2 = cmp_mask2 + cmp_mask1  # Combine masks

        # Apply the estimated mask to the complex spectrogram
        est_spec, est_wav, est_mask = self.apply_mask(cmp_spec, cmp_mask2)
        out_list.append(est_spec)
        out_list.append(est_wav)
        out_list.append(est_mask)
        return out_list
