# Copyright (c) 2024 Alibaba Inc (authors: Shengkui Zhao, Zexu Pan)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.backends.mps as mps
import soundfile as sf
import os
import subprocess
import librosa
from tqdm import tqdm
import numpy as np
from pydub import AudioSegment
from .utils.decode import decode_one_audio
from .utils.decode_batch import decode_one_audio_batch
from .dataloader.dataloader import DataReader

MAX_WAV_VALUE = 32768.0

class SpeechModel:
    """
    The SpeechModel class is a base class designed to handle speech processing tasks,
    such as loading, processing, and decoding audio data. It initializes the computational 
    device (CPU or GPU) and holds model-related attributes. The class is flexible and intended 
    to be extended by specific speech models for tasks like speech enhancement, speech separation, 
    target speaker extraction etc.

    Attributes:
    - args: Argument parser object that contains configuration settings.
    - device: The device (CPU or GPU) on which the model will run.
    - model: The actual model used for speech processing tasks (to be loaded by subclasses).
    - name: A placeholder for the model's name.
    - data: A dictionary to store any additional data related to the model, such as audio input.
    """

    def __init__(self, args):
        """
        Initializes the SpeechModel class by determining the computation device 
        (GPU or CPU) to be used for running the model, based on system availability.

        Args:
        - args: Argument parser object containing settings like whether to use CUDA (GPU) or not.
        """
        # Check if mps is available
        if mps.is_available():
            args.use_cuda = 1
            self.device = torch.device('cpu') #torch.device('mps')
        # Check if a GPU is available
        elif torch.cuda.is_available():
            # Find the GPU with the most free memory using a custom method
            free_gpu_id = self.get_free_gpu()
            if free_gpu_id is not None:
                args.use_cuda = 1
                torch.cuda.set_device(free_gpu_id)
                self.device = torch.device(f'cuda:{free_gpu_id}')
            else:
                # If no GPU is detected, use the CPU
                #print("No GPU found. Using CPU.")
                args.use_cuda = 0
                self.device = torch.device('cpu')
        else:
            # If no GPU is detected, use the CPU
            args.use_cuda = 0
            self.device = torch.device('cpu')

        self.args = args
        self.model = None
        self.name = None
        self.data = {}
        self.print = False

    def get_free_gpu(self):
        """
        Identifies the GPU with the most free memory using 'nvidia-smi' and returns its index.

        This function queries the available GPUs on the system and determines which one has 
        the highest amount of free memory. It uses the `nvidia-smi` command-line tool to gather 
        GPU memory usage data. If successful, it returns the index of the GPU with the most free memory.
        If the query fails or an error occurs, it returns None.

        Returns:
        int: Index of the GPU with the most free memory, or None if no GPU is found or an error occurs.
        """
        try:
            # Run nvidia-smi to query GPU memory usage and free memory
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
            gpu_info = result.stdout.decode('utf-8').strip().split('\n')

            free_gpu = None
            max_free_memory = 0
            for i, info in enumerate(gpu_info):
                used, free = map(int, info.split(','))
                if free > max_free_memory:
                    max_free_memory = free
                    free_gpu = i
            return free_gpu
        except Exception as e:
            print(f"Error finding free GPU: {e}")
            return None

    def download_model(self, model_name):
        checkpoint_dir = self.args.checkpoint_dir
        from huggingface_hub import snapshot_download
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print(f'downloading checkpoint for {model_name}')
        try:
            snapshot_download(repo_id=f'alibabasglab/{model_name}', local_dir=checkpoint_dir)
            return True
        except:
            return False
            
    def load_model(self):
        """
        Loads a pre-trained model checkpoints from a specified directory. It checks for
        the best model ('last_best_checkpoint') in the checkpoint directory. If a model is
        found, it loads the model state into the current model instance.

        If no checkpoint is found, it will try to download the model from huggingface.
        If the downloading fails, it prints a warning message.

        Steps:
        - Search for the best model checkpoint or the most recent one.
        - Load the model's state dictionary from the checkpoint file.

        Raises:
        - FileNotFoundError: If neither 'last_best_checkpoint' nor 'last_checkpoint' files are found.
        """
        # Define paths for the best model and the last checkpoint
        best_name = os.path.join(self.args.checkpoint_dir, 'last_best_checkpoint')
        # Check if the last best checkpoint exists
        if not os.path.isfile(best_name):
            if not self.download_model(self.name):
                # If downloading is unsuccessful
                print(f'Warning: Downloading model {self.name} is not successful. Please try again or manually download from https://huggingface.co/alibabasglab/{self.name}/tree/main !')
                return

        if isinstance(self.model, nn.ModuleList):
            with open(best_name, 'r') as f:
                model_name = f.readline().strip()
                checkpoint_path = os.path.join(self.args.checkpoint_dir, model_name)
                self._load_model(self.model[0], checkpoint_path, model_key='mossformer')
                model_name = f.readline().strip()
                checkpoint_path = os.path.join(self.args.checkpoint_dir, model_name)
                self._load_model(self.model[1], checkpoint_path, model_key='generator')
        else:
            # Read the model's checkpoint name from the file
            with open(best_name, 'r') as f:
                model_name = f.readline().strip()
            # Form the full path to the model's checkpoint
            checkpoint_path = os.path.join(self.args.checkpoint_dir, model_name)
            self._load_model(self.model, checkpoint_path, model_key='model')

    def _load_model(self, model, checkpoint_path, model_key=None):
        # Load the checkpoint file into memory (map_location ensures compatibility with different devices)
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        # Load the model's state dictionary (weights and biases) into the current model
        if model_key in checkpoint:
            pretrained_model = checkpoint[model_key]
        else:
            pretrained_model = checkpoint
        state = model.state_dict()
        for key in state.keys():
            if key in pretrained_model and state[key].shape == pretrained_model[key].shape:
                state[key] = pretrained_model[key]
            elif key.replace('module.', '') in pretrained_model and state[key].shape == pretrained_model[key.replace('module.', '')].shape:
                 state[key] = pretrained_model[key.replace('module.', '')]
            elif 'module.'+key in pretrained_model and state[key].shape == pretrained_model['module.'+key].shape:
                 state[key] = pretrained_model['module.'+key]
            elif self.print: print(f'{key} not loaded')
        model.load_state_dict(state)

    def decode(self):
        """
        Decodes the input audio data using the loaded model and ensures the output matches the original audio length.

        This method processes the audio through a speech model (e.g., for enhancement, separation, etc.),
        and truncates the resulting audio to match the original input's length. The method supports multiple speakers 
        if the model handles multi-speaker audio.

        Returns:
        output_audio: The decoded audio after processing, truncated to the input audio length. 
                  If multi-speaker audio is processed, a list of truncated audio outputs per speaker is returned.
        """
        # Decode the audio using the loaded model on the given device (e.g., CPU or GPU)
        output_audios = []
        for i in range(len(self.data['audio'])):
            output_audio = decode_one_audio(self.model, self.device, self.data['audio'][i], self.args)
            # Ensure the decoded output matches the length of the input audio
            if isinstance(output_audio, list):
                # If multi-speaker audio (a list of outputs), truncate each speaker's audio to input length
                for spk in range(self.args.num_spks):
                    output_audio[spk] = output_audio[spk][:self.data['audio_len']]
            else:
                # Single output, truncate to input audio length
                output_audio = output_audio[:self.data['audio_len']]
            output_audios.append(output_audio)
            
        if isinstance(output_audios[0], list):
            output_audios_np = []
            for spk in range(self.args.num_spks):
                output_audio_buf = []
                for i in range(len(output_audios)):
                    output_audio_buf.append(output_audios[i][spk])
                    #output_audio_buf = np.vstack((output_audio_buf, output_audios[i][spk])).T
                output_audios_np.append(np.array(output_audio_buf))
        else:
            output_audios_np = np.array(output_audios)
        return output_audios_np

    def decode_data(self, input_data):
        """
        Decodes the input audio data using the loaded model and ensures the output matches the original audio length.

        This method processes the audio through a speech model (e.g., for enhancement, separation, etc.),
        and truncates the resulting audio to match the original input's length. The method supports multiple speakers 
        if the model handles multi-speaker audio.

        Returns:
        output_audio: The decoded audio after processing, truncated to the input audio length. 
                  If multi-speaker audio is processed, a list of truncated audio outputs per speaker is returned.
        """
        # Decode the audio using the loaded model on the given device (e.g., CPU or GPU)
        output_audio = decode_one_audio_batch(self.model, self.device, input_data, self.args)
        return output_audio
       
    def process(self, input_path, online_write=False, output_path=None):
        """
        Load and process audio files from the specified input path. Optionally, 
        write the output audio files to the specified output directory.
        
        Args:
            input_path (str): Path to the input audio files or folder.
            online_write (bool): Whether to write the processed audio to disk in real-time.
            output_path (str): Optional path for writing output files. If None, output 
                               will be stored in self.result.
        
        Returns:
            dict or ndarray: Processed audio results either as a dictionary or as a single array, 
                             depending on the number of audio files processed. 
                             Returns None if online_write is enabled.
        """
        
        self.result = {}
        self.args.input_path = input_path
        data_reader = DataReader(self.args)  # Initialize a data reader to load the audio files


        # Check if online writing is enabled
        if online_write:
            output_wave_dir = self.args.output_dir  # Set the default output directory
            if isinstance(output_path, str):  # If a specific output path is provided, use it
                output_wave_dir = os.path.join(output_path, self.name)
            # Create the output directory if it does not exist
            if not os.path.isdir(output_wave_dir):
                os.makedirs(output_wave_dir)
        
        num_samples = len(data_reader)  # Get the total number of samples to process
        print(f'Running {self.name} ...')  # Display the model being used

        if self.args.task == 'target_speaker_extraction':
            from .utils.video_process import process_tse
            assert online_write == True
            process_tse(self.args, self.model, self.device, data_reader, output_wave_dir)
        else:
            # Disable gradient calculation for better efficiency during inference
            with torch.no_grad():
                for idx in tqdm(range(num_samples)):  # Loop over all audio samples
                    self.data = {}
                    # Read the audio, waveform ID, and audio length from the data reader
                    input_audio, wav_id, input_len, scalars, audio_info = data_reader[idx]
                    # Store the input audio and metadata in self.data
                    self.data['audio'] = input_audio
                    self.data['id'] = wav_id
                    self.data['audio_len'] = input_len
                    self.data.update(audio_info)
                    
                    # Perform the audio decoding/processing
                    output_audios = self.decode()

                    # Perform audio renormalization
                    if not isinstance(output_audios, list):
                        if len(scalars) > 1:
                            for i in range(len(scalars)):
                                output_audios[:,i] = output_audios[:,i] * scalars[i]
                        else:
                                output_audios = output_audios * scalars[0]
                        
                    if online_write:
                        # If online writing is enabled, save the output audio to files
                        if isinstance(output_audios, list):
                            # In case of multi-speaker output, save each speaker's output separately
                            for spk in range(self.args.num_spks):
                                output_file = os.path.join(output_wave_dir, wav_id.replace('.'+self.data['ext'], f'_s{spk+1}.'+self.data['ext']))
                                self.write_audio(output_file, key=None, spk=spk, audio=output_audios)
                        else:
                            # Single-speaker or standard output
                            output_file = os.path.join(output_wave_dir, wav_id)
                            self.write_audio(output_file, key=None, spk=None, audio=output_audios)
                    else:
                        # If not writing to disk, store the output in the result dictionary
                        self.result[wav_id] = output_audios
            
            # Return the processed results if not writing to disk
            if not online_write:
                if len(self.result) == 1:
                    # If there is only one result, return it directly
                    return next(iter(self.result.values()))
                else:
                    # Otherwise, return the entire result dictionary
                    return self.result

    def write_audio(self, output_path, key=None, spk=None, audio=None):
        """
        This function writes an audio signal to an output file, applying necessary transformations
        such as resampling, channel handling, and format conversion based on the provided parameters
        and the instance's internal settings.
        
        Args:
            output_path (str): The file path where the audio will be saved.
            key (str, optional): The key used to retrieve audio from the internal result dictionary
                                  if audio is not provided.
            spk (str, optional): A specific speaker identifier, used to extract a particular speaker's
                                 audio from a multi-speaker dataset or result.
            audio (numpy.ndarray, optional): A numpy array containing the audio data to be written.
                                 If provided, key and spk are ignored.
        """
        
        if audio is not None:
            if spk is not None:
                result_ = audio[spk]
            else:
                result_ = audio
        else:
            if spk is not None:
                result_ = self.result[key][spk]
            else:
                result_ = self.result[key]
                
        if self.data['sample_rate'] != self.args.sampling_rate:
            if self.data['channels'] == 2:
                left_channel = librosa.resample(result_[0,:], orig_sr=self.args.sampling_rate, target_sr=self.data['sample_rate'])
                right_channel = librosa.resample(result_[1,:], orig_sr=self.args.sampling_rate, target_sr=self.data['sample_rate'])
                result = np.vstack((left_channel, right_channel)).T
            else:
                result = librosa.resample(result_[0,:], orig_sr=self.args.sampling_rate, target_sr=self.data['sample_rate'])
        else:
            if self.data['channels'] == 2:
                left_channel = result_[0,:]
                right_channel = result_[1,:]
                result = np.vstack((left_channel, right_channel)).T
            else:
                result = result_[0,:]
                
        if self.data['sample_width'] == 4: ##32 bit float
            MAX_WAV_VALUE = 2147483648.0
            np_type = np.int32
        elif self.data['sample_width'] == 2: ##16 bit int
            MAX_WAV_VALUE = 32768.0
            np_type = np.int16
        else:
            self.data['sample_width'] = 2 ##16 bit int
            MAX_WAV_VALUE = 32768.0
            np_type = np.int16
                        
        result = result * MAX_WAV_VALUE
        result = result.astype(np_type)
        audio_segment = AudioSegment(
            result.tobytes(),  # Raw audio data as bytes
            frame_rate=self.data['sample_rate'],  # Sample rate
            sample_width=self.data['sample_width'],          # No. bytes per sample
            channels=self.data['channels']               # No. channels
        )
        audio_format = 'ipod' if self.data['ext'] in ['m4a', 'aac'] else self.data['ext']
        audio_segment.export(output_path, format=audio_format)
                    
    def write(self, output_path, add_subdir=False, use_key=False):
        """
        Write the processed audio results to the specified output path.

        Args:
            output_path (str): The directory or file path where processed audio will be saved. If not 
                               provided, defaults to self.args.output_dir.
            add_subdir (bool): If True, appends the model name as a subdirectory to the output path.
            use_key (bool): If True, uses the result dictionary's keys (audio file IDs) for filenames.

        Returns:
            None: Outputs are written to disk, no data is returned.
        """

        # Ensure the output path is a string. If not provided, use the default output directory
        if not isinstance(output_path, str):
            output_path = self.args.output_dir

        # If add_subdir is enabled, create a subdirectory for the model name
        if add_subdir:
            if os.path.isfile(output_path):
                print(f'File exists: {output_path}, remove it and try again!')
                return
            output_path = os.path.join(output_path, self.name)
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
                
        # check if the output dir exists, if not then create it
        directory_path = os.path.dirname(output_path)
        if not os.path.isdir(directory_path):
            os.makedirs(directory_path)
            
        # Ensure proper directory setup when using keys for filenames
        if use_key and not os.path.isdir(output_path):
            if os.path.exists(output_path):
                print(f'File exists: {output_path}, remove it and try again!')
                return
            os.makedirs(output_path)
        # If not using keys and output path is a directory, check for conflicts
        if not use_key and os.path.isdir(output_path):
            print(f'Directory exists: {output_path}, remove it and try again!')
            return

        # Iterate over the results dictionary to write the processed audio to disk
        for key in self.result:
            if use_key:
                # If using keys, format filenames based on the result dictionary's keys (audio IDs)
                if isinstance(self.result[key], list):  # For multi-speaker outputs
                    for spk in range(self.args.num_spks):
                        output_file = os.path.join(output_path, key.replace('.'+self.data['ext'], f'_s{spk+1}.'+self.data['ext']))
                        self.write_audio(output_file, key, spk)
                else:
                    output_file = os.path.join(output_path, key)
                    self.write_audio(output_path, key)
            else:
                # If not using keys, write audio to the specified output path directly
                if isinstance(self.result[key], list):  # For multi-speaker outputs
                    for spk in range(self.args.num_spks):
                        output_file = output_path.replace('.'+self.data['ext'], f'_s{spk+1}.'+self.data['ext'])
                        self.write_audio(output_file, key, spk)
                else:
                    self.write_audio(output_path, key)
                    
# The model classes for specific sub-tasks

class CLS_FRCRN_SE_16K(SpeechModel):
    """
    A subclass of SpeechModel that implements a speech enhancement model using 
    the FRCRN architecture for 16 kHz speech enhancement.
    
    Args:
        args (Namespace): The argument parser containing model configurations and paths.
    """

    def __init__(self, args):
        # Initialize the parent SpeechModel class
        super(CLS_FRCRN_SE_16K, self).__init__(args)
        
        # Import the FRCRN speech enhancement model for 16 kHz
        from .models.frcrn_se.frcrn import FRCRN_SE_16K
        
        # Initialize the model
        self.model = FRCRN_SE_16K(args).model
        self.name = 'FRCRN_SE_16K'
        
        # Load pre-trained model checkpoint
        self.load_model()
        
        # Move model to the appropriate device (GPU/CPU)
        if args.use_cuda == 1:
            self.model.to(self.device)
        
        # Set the model to evaluation mode (no gradient calculation)
        self.model.eval()

class CLS_MossFormer2_SE_48K(SpeechModel):
    """
    A subclass of SpeechModel that implements the MossFormer2 architecture for 
    48 kHz speech enhancement.
    
    Args:
        args (Namespace): The argument parser containing model configurations and paths.
    """

    def __init__(self, args):
        # Initialize the parent SpeechModel class
        super(CLS_MossFormer2_SE_48K, self).__init__(args)
        
        # Import the MossFormer2 speech enhancement model for 48 kHz
        from .models.mossformer2_se.mossformer2_se_wrapper import MossFormer2_SE_48K
        
        # Initialize the model
        self.model = MossFormer2_SE_48K(args).model
        self.name = 'MossFormer2_SE_48K'
        
        # Load pre-trained model checkpoint
        self.load_model()
        
        # Move model to the appropriate device (GPU/CPU)
        if args.use_cuda == 1:
            self.model.to(self.device)
        
        # Set the model to evaluation mode (no gradient calculation)
        self.model.eval()

class CLS_MossFormer2_SR_48K(SpeechModel):
    """
    A subclass of SpeechModel that implements the MossFormer2 architecture for
    48 kHz speech super-resolution.
    
    Args:
        args (Namespace): The argument parser containing model configurations and paths.
    """

    def __init__(self, args):
        # Initialize the parent SpeechModel class
        super(CLS_MossFormer2_SR_48K, self).__init__(args)
        
        # Import the MossFormer2 speech enhancement model for 48 kHz
        from .models.mossformer2_sr.mossformer2_sr_wrapper import MossFormer2_SR_48K
        
        # Initialize the model
        self.model = nn.ModuleList()
        self.model.append(MossFormer2_SR_48K(args).model_m)
        self.model.append(MossFormer2_SR_48K(args).model_g)
        self.name = 'MossFormer2_SR_48K'
        
        # Load pre-trained model checkpoint
        self.load_model()
        
        # Move model to the appropriate device (GPU/CPU)
        if args.use_cuda == 1:
            for model in self.model:
                model.to(self.device)
        
        # Set the model to evaluation mode (no gradient calculation)
        for model in self.model:
            model.eval()
        self.model[1].remove_weight_norm()

class CLS_MossFormerGAN_SE_16K(SpeechModel):
    """
    A subclass of SpeechModel that implements the MossFormerGAN architecture for 
    16 kHz speech enhancement, utilizing GAN-based speech processing.
    
    Args:
        args (Namespace): The argument parser containing model configurations and paths.
    """

    def __init__(self, args):
        # Initialize the parent SpeechModel class
        super(CLS_MossFormerGAN_SE_16K, self).__init__(args)
        
        # Import the MossFormerGAN speech enhancement model for 16 kHz
        from .models.mossformer_gan_se.generator import MossFormerGAN_SE_16K
        
        # Initialize the model
        self.model = MossFormerGAN_SE_16K(args).model
        self.name = 'MossFormerGAN_SE_16K'
        
        # Load pre-trained model checkpoint
        self.load_model()
        
        # Move model to the appropriate device (GPU/CPU)
        if args.use_cuda == 1:
            self.model.to(self.device)
        
        # Set the model to evaluation mode (no gradient calculation)
        self.model.eval()

class CLS_MossFormer2_SS_16K(SpeechModel):
    """
    A subclass of SpeechModel that implements the MossFormer2 architecture for 
    16 kHz speech separation.
    
    Args:
        args (Namespace): The argument parser containing model configurations and paths.
    """

    def __init__(self, args):
        # Initialize the parent SpeechModel class
        super(CLS_MossFormer2_SS_16K, self).__init__(args)
        
        # Import the MossFormer2 speech separation model for 16 kHz
        from .models.mossformer2_ss.mossformer2 import MossFormer2_SS_16K
        
        # Initialize the model
        self.model = MossFormer2_SS_16K(args).model
        self.name = 'MossFormer2_SS_16K'
        
        # Load pre-trained model checkpoint
        self.load_model()
        
        # Move model to the appropriate device (GPU/CPU)
        if args.use_cuda == 1:
            self.model.to(self.device)
        
        # Set the model to evaluation mode (no gradient calculation)
        self.model.eval()


class CLS_AV_MossFormer2_TSE_16K(SpeechModel):
    """
    A subclass of SpeechModel that implements an audio-visual (AV) model using 
    the AV-MossFormer2 architecture for target speaker extraction (TSE) at 16 kHz. 
    This model leverages both audio and visual cues to perform speaker extraction.
    
    Args:
        args (Namespace): The argument parser containing model configurations and paths.
    """

    def __init__(self, args):
        # Initialize the parent SpeechModel class
        super(CLS_AV_MossFormer2_TSE_16K, self).__init__(args)
        
        # Import the AV-MossFormer2 model for 16 kHz target speech enhancement
        from .models.av_mossformer2_tse.av_mossformer2 import AV_MossFormer2_TSE_16K
        
        # Initialize the model
        self.model = AV_MossFormer2_TSE_16K(args).model
        self.name = 'AV_MossFormer2_TSE_16K'
        
        # Load pre-trained model checkpoint
        self.load_model()
        
        # Move model to the appropriate device (GPU/CPU)
        if args.use_cuda == 1:
            self.model.to(self.device)
        
        # Set the model to evaluation mode (no gradient calculation)
        self.model.eval()


