import streamlit as st
from clearvoice import ClearVoice
import os
import time
import numpy as np
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, collect_chunks
import soundfile as sf

st.set_page_config(page_title="ClearerVoice Studio", layout="wide")
temp_dir = 'temp'

# Preload VAD model
vad_model = load_silero_vad()

def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        # Check if temp directory exists, create if not
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        # Save to temp directory, overwrite if file exists
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        return temp_path
    return None

def main():
    st.title("ClearerVoice Studio")
    
    tabs = st.tabs(["Speech Enhancement", "Speech Separation", "Target Speaker Extraction"])
    
    with tabs[0]:
        st.header("Speech Enhancement")
        
        # Model selection
        se_models = ['MossFormer2_SE_48K', 'FRCRN_SE_16K', 'MossFormerGAN_SE_16K']
        selected_model = st.selectbox("Select Model", se_models)
        
        # VAD option
        enable_vad = st.checkbox("Enable VAD Preprocessing", key='se_vad')
        
        # File upload
        uploaded_file = st.file_uploader("Upload Audio File", type=['wav'], key='se')
        
        if st.button("Start Processing", key='se_process'):
            if uploaded_file is not None:
                with st.spinner('Processing...'):
                    # Start time
                    start_time = time.time()
                    
                    # Save uploaded file
                    input_path = save_uploaded_file(uploaded_file)
                    
                    if enable_vad:
                        # Read original audio and sample rate
                        orig_audio, orig_sr = sf.read(input_path)
                        if orig_audio.ndim > 1:
                            orig_audio = orig_audio.mean(axis=1)
                            
                        # Use Silero VAD (always works at 16kHz)
                        vad_wav = read_audio(input_path)
                        timestamps = get_speech_timestamps(vad_wav, vad_model, sampling_rate=16000)
                        
                        # Map timestamps to original sample rate
                        scale = orig_sr / 16000.0
                        speech_ts = [{'start': int(ts['start'] * scale), 
                                       'end': int(ts['end'] * scale)} 
                                     for ts in timestamps]
                        
                        # Extract speech segments at original sample rate
                        segs = [orig_audio[d['start']:d['end']] for d in speech_ts]
                        speech_input = np.concatenate(segs) if segs else np.array([])
                        
                        # Save speech segments to temp file
                        temp_seg_path = os.path.join(temp_dir, f"speech_segs_{uploaded_file.name}")
                        sf.write(temp_seg_path, speech_input, orig_sr)
                        model_input = temp_seg_path
                    else:
                        # Process without VAD
                        model_input = input_path
                        
                    # Initialize ClearVoice and process audio
                    myClearVoice = ClearVoice(task='speech_enhancement', 
                                             model_names=[selected_model])
                    output_wav = myClearVoice(input_path=model_input, online_write=False)
                    
                    # Prepare output directory
                    output_dir = os.path.join(temp_dir, "speech_enhancement_output")
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, 
                                              f"output_{selected_model}{'_vad' if enable_vad else ''}.wav")
                    
                    if enable_vad:
                        # Get model's internal sample rate
                        model_sr = myClearVoice.models[0].args.sampling_rate
                        enhanced = np.squeeze(output_wav)
                        
                        # Resample back to original rate if needed
                        if model_sr != orig_sr and enhanced.size:
                            import librosa
                            enhanced = librosa.resample(enhanced, 
                                                        orig_sr=model_sr, 
                                                        target_sr=orig_sr)
                        
                        # Reconstruct full audio by placing enhanced segments
                        full_enh = np.zeros_like(orig_audio)
                        idx = 0
                        for d in speech_ts:
                            s, e = d['start'], d['end']
                            length = e - s
                            full_enh[s:e] = enhanced[idx: idx + length]
                            idx += length
                            
                        # Write output with original sample rate
                        sf.write(output_path, full_enh, orig_sr, format='WAV', subtype='PCM_16')
                    else:
                        # Use ClearVoice's native writer for non-VAD mode
                        myClearVoice.write(output_wav, output_path=output_path)
                    
                    # Display results
                    process_time = time.time() - start_time
                    st.success(f"Processing completed in {process_time:.2f} seconds")
                    st.audio(output_path)
            else:
                st.error("Please upload an audio file first")
    
    with tabs[1]:
        st.header("Speech Separation")
        
        # File upload
        uploaded_file = st.file_uploader("Upload Mixed Audio File", type=['wav', 'avi'], key='ss')
        
        if st.button("Start Separation", key='ss_process'):
            if uploaded_file is not None:
                with st.spinner('Processing...'):
                    # Start time
                    start_time = time.time()
                    
                    # Save uploaded file
                    input_path = save_uploaded_file(uploaded_file)

                    # Extract audio if input is video file
                    if input_path.endswith(('.avi')):
                        import cv2
                        video = cv2.VideoCapture(input_path)
                        audio_path = input_path.replace('.avi','.wav')
                        
                        # Extract audio
                        import subprocess
                        cmd = f"ffmpeg -i {input_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
                        subprocess.call(cmd, shell=True)
                        
                        input_path = audio_path
                    
                    # Initialize ClearVoice
                    myClearVoice = ClearVoice(task='speech_separation', 
                                            model_names=['MossFormer2_SS_16K'])
                    
                    # Process audio
                    output_wav = myClearVoice(input_path=input_path, 
                                            online_write=False)
                    
                    output_dir = os.path.join(temp_dir, "speech_separation_output")
                    os.makedirs(output_dir, exist_ok=True)

                    file_name = os.path.basename(input_path).split('.')[0]
                    base_file_name = 'output_MossFormer2_SS_16K_'
                    
                    # Save processed audio
                    output_path = os.path.join(output_dir, f"{base_file_name}{file_name}.wav")
                    myClearVoice.write(output_wav, output_path=output_path)
                    
                    # Display results
                    process_time = time.time() - start_time
                    st.success(f"Processing completed in {process_time:.2f} seconds")
                    st.text(output_dir)

            else:
                st.error("Please upload an audio file first")
    
    with tabs[2]:
        st.header("Target Speaker Extraction")
        
        # File upload
        uploaded_file = st.file_uploader("Upload Video File", type=['mp4', 'avi'], key='tse')
        
        if st.button("Start Extraction", key='tse_process'):
            if uploaded_file is not None:
                with st.spinner('Processing...'):
                    # Start time
                    start_time = time.time()
                    
                    # Save uploaded file
                    input_path = save_uploaded_file(uploaded_file)
                    
                    # Create output directory
                    output_dir = os.path.join(temp_dir, "videos_tse_output")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Initialize ClearVoice
                    myClearVoice = ClearVoice(task='target_speaker_extraction', 
                                            model_names=['AV_MossFormer2_TSE_16K'])
                    
                    # Process video
                    myClearVoice(input_path=input_path, 
                                online_write=True,
                                output_path=output_dir)
                    
                    # Display results
                    process_time = time.time() - start_time
                    st.success(f"Processing completed in {process_time:.2f} seconds")
                    
                    # Display output folder
                    st.subheader("Output Folder")
                    st.text(output_dir)
                
            else:
                st.error("Please upload a video file first")

if __name__ == "__main__":    
    main()