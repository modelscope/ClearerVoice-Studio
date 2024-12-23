from clearvoice import ClearVoice  # Import the ClearVoice class for speech processing tasks

if __name__ == '__main__':
    ## ----------------- Demo One: Using a Single Model ----------------------
    if False:  # This block demonstrates how to use a single model for speech enhancement
        # Initialize ClearVoice for the task of speech enhancement using the MossFormer2_SE_48K model
        myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])

        # 1st calling method: 
        #   Process an input waveform and return the enhanced output waveform
        # - input_path (str): Path to the input noisy audio file (input.wav)
        # - output_wav (dict or ndarray) : The enhanced output waveform
        output_wav = myClearVoice(input_path='samples/input.wav')
        # Write the processed waveform to an output file
        # - output_path (str): Path to save the enhanced audio file (output_MossFormer2_SE_48K.wav)
        myClearVoice.write(output_wav, output_path='samples/output_MossFormer2_SE_48K.wav')

        # 2nd calling method: 
        #   Process and write audio files directly
        # - input_path (str): Path to the directory of input noisy audio files
        # - online_write (bool): Set to True to enable saving the enhanced audio directly to files during processing
        # - output_path (str): Path to the directory to save the enhanced output files
        myClearVoice(input_path='samples/path_to_input_wavs', online_write=True, output_path='samples/path_to_output_wavs')

        # 3rd calling method: 
        #   Use an .scp file to specify input audio paths
        # - input_path (str): Path to a .scp file listing multiple audio file paths
        # - online_write (bool): Set to True to enable saving the enhanced audio directly to files during processing
        # - output_path (str): Path to the directory to save the enhanced output files
        myClearVoice(input_path='samples/scp/audio_samples.scp', online_write=True, output_path='samples/path_to_output_wavs_scp')


    ## ---------------- Demo Two: Using Multiple Models -----------------------
    if True:  # This block demonstrates how to use multiple models for speech enhancement
        # Initialize ClearVoice for the task of speech enhancement using two models: MossFormer2_SE_48K and FRCRN_SE_16K
        myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K', 'FRCRN_SE_16K'])

        # 1st calling method: 
        #   Process an input waveform using the multiple models and return the enhanced output waveform
        # - input_path (str): Path to the input noisy audio file (input.wav)
        # - output_wav (dict or ndarray) : The returned output waveforms after being processed by the models
        output_wav = myClearVoice(input_path='samples/input.wav')
        # Write the processed waveform to an output file
        # - output_path (str): Path to the directory to save the enhanced audio file using the same file name as input (input.wav)
        myClearVoice.write(output_wav, output_path='samples/path_to_output_wavs')

        # 2nd calling method: 
        #   Process and write audio files directly using multiple models
        # - input_path (str): Path to the directory of input noisy audio files
        # - online_write (bool): Set to True to enable saving the enhanced audio directly to files during processing
        # - output_path (str): Path to the directory to save the enhanced output files
        myClearVoice(input_path='samples/path_to_input_wavs', online_write=True, output_path='samples/path_to_output_wavs')

        # 3rd calling method: 
        #   Use an .scp file to specify input audio paths for multiple models
        # - input_path (str): Path to a .scp file listing multiple audio file paths
        # - online_write (bool): Set to True to enable saving the enhanced output during processing
        # - output_path (str): Path to the directory to save the enhanced output files
        myClearVoice(input_path='samples/scp/audio_samples.scp', online_write=True, output_path='samples/path_to_output_wavs_scp')

    ## ---------------- Demo Three: Real-Time Processing -----------------------
    if False:  # This block demonstrates how to use the FRCRN_SE_16K model for real-time speech enhancement
        # Initialize ClearVoice for the task of speech enhancement using the FRCRN_SE_16K model
        myClearVoice = ClearVoice(task='speech_enhancement', model_names=['FRCRN_SE_16K'])

        # 1st calling method: 
        #   Process an input waveform in real-time and return the enhanced output waveform
        # - input_path (str): Path to the input noisy audio file (input_realtime.wav)
        # - output_wav (dict or ndarray) : The enhanced output waveform
        output_wav = myClearVoice(input_path='samples/input_realtime.wav', online_write=False)
        # Write the processed waveform to an output file
        # - output_path (str): Path to save the enhanced audio file (output_FRCRN_SE_16K_realtime.wav)
        myClearVoice.write(output_wav, output_path='samples/output_FRCRN_SE_16K_realtime.wav')

        # 2nd calling method: 
        #   Process and write audio files directly in real-time
        # - input_path (str): Path to the directory of input noisy audio files
        # - online_write (bool): Set to True to enable saving the enhanced audio directly to files during processing
        # - output_path (str): Path to the directory to save the enhanced output files
        myClearVoice(input_path='samples/path_to_input_wavs_realtime', online_write=True, output_path='samples/path_to_output_wavs_realtime')

        # 3rd calling method: 
        #   Use an .scp file to specify input audio paths for real-time processing
        # - input_path (str): Path to a .scp file listing multiple audio file paths
        # - online_write (bool): Set to True to enable saving the enhanced audio directly to files during processing
        # - output_path (str): Path to the directory to save the enhanced output files
        myClearVoice(input_path='samples/scp/audio_samples_realtime.scp', online_write=True, output_path='samples/path_to_output_wavs_realtime_scp')
