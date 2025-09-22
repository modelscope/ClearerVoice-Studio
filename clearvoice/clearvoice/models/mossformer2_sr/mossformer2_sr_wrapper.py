from .generator import Mossformer, Generator
import torch.nn as nn

class MossFormer2_SR_48K(nn.Module):
    """
    The MossFormer2_SR_48K model for speech super-resolution.

    This class encapsulates the functionality of the MossFormer2 and HiFi-Gan
    Generator within a higher-level model. It processes input audio data to produce
    higher-resolution outputs.

    Arguments
    ---------
    args : Namespace
        Configuration arguments that may include hyperparameters 
        and model settings (not utilized in this implementation but 
        can be extended for flexibility).

    Example
    ---------
    >>> model = MossFormer2_SR_48K(args).model
    >>> x = torch.randn(10, 180, 2000)  # Example input
    >>> outputs = model(x)  # Forward pass
    >>> outputs.shape, mask.shape  # Check output shapes
    """

    def __init__(self, args):
        super(MossFormer2_SR_48K, self).__init__()
        # Initialize the TestNet model, which contains the MossFormer MaskNet
        self.model_m = Mossformer()  # Instance of TestNet
        self.model_g = Generator(args)

    def forward(self, x):
        """
        Forward pass through the model.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, S], where B is the batch size,
            N is the number of mel bins (80 in this case), and S is the
            sequence length (e.g., time frames).

        Returns
        -------
        outputs : torch.Tensor
            Bandwidth expanded audio output tensor from the model.

        """
        x = self.model_m(x)  # Get outputs and mask from TestNet
        return self.model_g(x)
