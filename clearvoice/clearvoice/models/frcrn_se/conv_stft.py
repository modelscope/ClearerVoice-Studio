import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.signal import get_window
import scipy

def init_kernels(win_len, win_inc, fft_len, win_type=None, invers=False):
    """
    Initialize the kernels for STFT and iSTFT operations.

    This function generates the kernel for the convolutional layers used in the short-time Fourier transform (STFT)
    and its inverse (iSTFT). The kernel is created based on the window type and length specified.

    Args:
        win_len (int): Length of the window.
        win_inc (int): Window increment (hop length).
        fft_len (int): Length of the FFT.
        win_type (str, optional): Type of window to apply (e.g., 'hanning', 'hamming'). Default is None (rectangular window).
        invers (bool, optional): If True, computes the pseudo-inverse of the kernel. Default is False.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The kernel used for convolution, with shape (2 * win_len, 1, fft_len).
            - torch.Tensor: The window applied to the kernel, with shape (1, win_len, 1).
    """
    if win_type == 'None' or win_type is None:
        window = np.ones(win_len)
    else:
        # Convert 'hanning' to 'hann' if using scipy version 1.10.1 or higher
        if scipy.__version__ >= '1.10.1' and win_type == 'hanning':
            win_type = 'hann'
        window = get_window(win_type, win_len, fftbins=True)**0.5
    
    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]  # Compute Fourier basis for the identity matrix
    real_kernel = np.real(fourier_basis)  # Extract real part
    imag_kernel = np.imag(fourier_basis)  # Extract imaginary part
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T  # Combine real and imaginary parts

    if invers:
        kernel = np.linalg.pinv(kernel).T  # Compute pseudo-inverse if required

    kernel = kernel * window  # Apply window to kernel
    kernel = kernel[:, None, :]  # Add singleton dimension for compatibility
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None,:,None].astype(np.float32))


class ConvSTFT(nn.Module):
    """
    Convolutional layer that performs Short-Time Fourier Transform (STFT).

    This class applies the STFT to input signals using convolution with pre-computed kernels.
    It can return either the complex STFT representation or the magnitude and phase components.

    Attributes:
        weight (nn.Parameter): Learnable convolution kernel for STFT.
        feature_type (str): Specifies whether to return 'complex' or 'real' features.
        stride (int): The stride used for convolution, typically equal to win_inc.
        win_len (int): The length of the window used in STFT.
        dim (int): The FFT length, determining the number of output features.
    """

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        """
        Initializes the ConvSTFT layer.

        Args:
            win_len (int): Length of the window.
            win_inc (int): Window increment (hop length).
            fft_len (int, optional): Length of the FFT. If None, it's computed based on win_len.
            win_type (str, optional): Type of window to use (default is 'hamming').
            feature_type (str, optional): Specifies the output feature type ('real' or 'complex'). Default is 'real'.
            fix (bool, optional): If True, the kernel weights are fixed and not learnable. Default is True.
        """
        super(ConvSTFT, self).__init__()

        if fft_len is None:
            self.fft_len = 1 << (win_len - 1).bit_length()  # Calculate fft_len based on win_len
        else:
            self.fft_len = fft_len
        
        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)  # Initialize STFT kernel
        self.weight = nn.Parameter(kernel, requires_grad=(not fix))  # Create a learnable parameter for the kernel
        self.feature_type = feature_type
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs):
        """
        Forward pass through the ConvSTFT layer.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, channels, length).

        Returns:
            tuple or torch.Tensor: Depending on feature_type, returns either:
                - torch.Tensor: The complex STFT output if feature_type is 'complex'.
                - tuple: A tuple containing the magnitude and phase tensors if feature_type is 'real'.
        """
        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs, 1)  # Add channel dimension if not present

        outputs = F.conv1d(inputs, self.weight, stride=self.stride)  # Perform convolution to compute STFT
         
        if self.feature_type == 'complex':
            return outputs
        else:
            dim = self.dim // 2 + 1  # Calculate the size for the real and imaginary components
            real = outputs[:, :dim, :]  # Extract real part
            imag = outputs[:, dim:, :]  # Extract imaginary part
            mags = torch.sqrt(real**2 + imag**2)  # Compute magnitude
            phase = torch.atan2(imag, real)  # Compute phase
            return mags, phase  # Return magnitude and phase


class ConviSTFT(nn.Module):
    """
    Convolutional layer that performs Inverse Short-Time Fourier Transform (iSTFT).

    This class applies the iSTFT to reconstruct the time-domain signal from the frequency-domain representation
    obtained from the ConvSTFT layer.

    Attributes:
        weight (nn.Parameter): Learnable convolution kernel for iSTFT.
        feature_type (str): Specifies whether to use 'real' or 'complex' features for reconstruction.
        win_type (str): Type of window used during iSTFT.
        win_len (int): The length of the window used in iSTFT.
        win_inc (int): The window increment (hop length).
        stride (int): The stride used for transposed convolution, typically equal to win_inc.
        dim (int): The FFT length, determining the number of output features.
        window (torch.Tensor): Buffer for the window used in iSTFT.
        enframe (torch.Tensor): Buffer for the framing matrix.
    """

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        """
        Initializes the ConviSTFT layer.

        Args:
            win_len (int): Length of the window.
            win_inc (int): Window increment (hop length).
            fft_len (int, optional): Length of the FFT. If None, it's computed based on win_len.
            win_type (str, optional): Type of window to use (default is 'hamming').
            feature_type (str, optional): Specifies the output feature type ('real' or 'complex'). Default is 'real'.
            fix (bool, optional): If True, the kernel weights are fixed and not learnable. Default is True.
        """
        super(ConviSTFT, self).__init__()

        if fft_len is None:
            self.fft_len = int(2**np.ceil(np.log2(win_len)))  # Calculate fft_len based on win_len
        else:
            self.fft_len = fft_len

        kernel, window = init_kernels(win_len, win_inc, self.fft_len, win_type, invers=True)  # Initialize iSTFT kernel
        self.weight = nn.Parameter(kernel, requires_grad=(not fix))  # Create a learnable parameter for the kernel
        self.feature_type = feature_type
        self.win_type = win_type
        self.win_len = win_len
        self.win_inc = win_inc
        self.stride = win_inc
        self.dim = self.fft_len
        self.register_buffer('window', window)  # Register the window as a buffer
        self.register_buffer('enframe', torch.eye(win_len)[:, None, :])  # Framing matrix for overlap-add method

    def forward(self, inputs, phase=None):
        """
        Forward pass through the ConviSTFT layer.

        Args:
            inputs (torch.Tensor): Input tensor of shape [B, N+2, T] for complex spectra 
                                   or [B, N//2+1, T] for magnitude spectra.
            phase (torch.Tensor, optional): Phase tensor of shape [B, N//2+1, T]. If provided, used to reconstruct the complex spectra.

        Returns:
            torch.Tensor: Reconstructed time-domain signal.
        """
        if phase is not None:
            # Reconstruct real and imaginary components from magnitude and phase
            real = inputs * torch.cos(phase)  # Real part
            imag = inputs * torch.sin(phase)  # Imaginary part
            inputs = torch.cat([real, imag], 1)  # Concatenate to form complex input

        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride)  # Perform transposed convolution for iSTFT

        # Compute the overlap-add normalization
        t = self.window.repeat(1, 1, inputs.size(-1))**2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)  # Apply the framing matrix for overlap-add
        outputs = outputs / (coff + 1e-8)  # Normalize the output to prevent division by zero
        return outputs


def test_fft():
    """
    Test the ConvSTFT layer against a NumPy STFT reference.

    This function generates a random input signal and computes its STFT using the ConvSTFT layer,
    then compares the output with a reference implementation that uses the same win_len-based framing.
    """
    torch.manual_seed(20)
    win_len = 320
    win_inc = 160 
    fft_len = 512
    inputs = torch.randn([1, 1, 16000*4])  # Random input tensor
    fft = ConvSTFT(win_len, win_inc, fft_len, win_type='hanning', feature_type='real')  # Initialize ConvSTFT

    outputs1 = fft(inputs)[0]  # Compute STFT using ConvSTFT
    outputs1 = outputs1.numpy()[0]  # Convert to NumPy array for comparison
    np_inputs = inputs.numpy().reshape([-1])  # Reshape input for Librosa
    window = get_window('hann', win_len, fftbins=True)**0.5
    frames = np.stack([np_inputs[start:start + win_len] for start in range(0, len(np_inputs) - win_len + 1, win_inc)])
    reference_stft = np.fft.rfft(frames * window[None, :], n=fft_len, axis=1).T
    print(np.mean((outputs1 - np.abs(reference_stft))**2))  # Print mean squared error between the two STFT outputs


def test_ifft1():
    """
    Test the ConviSTFT layer by reconstructing a waveform from the STFT output.

    This function reads an audio file, applies the ConvSTFT to compute its STFT, and then
    uses the ConviSTFT to reconstruct the time-domain signal. The reconstructed signal is saved to a file
    and compared to the original to evaluate the reconstruction accuracy.
    """
    import soundfile as sf
    N = 100
    inc = 75
    fft_len = 512
    torch.manual_seed(N)

    # Read input audio file and reshape
    data = sf.read('../../wavs/ori.wav')[0]
    inputs = data.reshape([1, 1, -1])  # Reshape to [1, 1, length]

    fft = ConvSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')  # Initialize ConvSTFT
    ifft = ConviSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')  # Initialize ConviSTFT
    
    inputs = torch.from_numpy(inputs.astype(np.float32))  # Convert to torch tensor
    outputs1 = fft(inputs)  # Compute STFT
    outputs2 = ifft(outputs1)  # Reconstruct waveform from STFT
    sf.write('conv_stft.wav', outputs2.numpy()[0, 0, :], 16000)  # Save reconstructed waveform to file
    print('wav MSE', torch.mean(torch.abs(inputs[..., :outputs2.size(2)] - outputs2)))  # Print mean squared error


def test_ifft2():
    """
    Test the iSTFT reconstruction from a random input signal.

    This function generates a random signal, computes its STFT, and then reconstructs it using the ConviSTFT layer.
    The reconstructed waveform is saved to a file, and the mean squared error is printed to evaluate accuracy.
    """
    N = 400
    inc = 100
    fft_len = 512
    np.random.seed(20)
    torch.manual_seed(20)
    
    # Generate a random signal
    t = np.random.randn(16000*4) * 0.005
    t = np.clip(t, -1, 1)  # Clip to [-1, 1] range
    input = torch.from_numpy(t[None, None, :].astype(np.float32))  # Reshape for input

    fft = ConvSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')  # Initialize ConvSTFT
    ifft = ConviSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')  # Initialize ConviSTFT
    
    out1 = fft(input)  # Compute STFT
    output = ifft(out1)  # Reconstruct waveform from STFT
    print('random MSE', torch.mean(torch.abs(input - output)**2))  # Print mean squared error
    import soundfile as sf
    sf.write('zero.wav', output[0, 0].numpy(), 16000)  # Save reconstructed waveform to file


if __name__ == '__main__':
    #test_fft()
    test_ifft1()  # Run the iSTFT reconstruction test
    #test_ifft2()

