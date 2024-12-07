import streamlit as st
from clearvoice import ClearVoice
import os
import tempfile

st.set_page_config(page_title="ClearerVoice Studio", layout="wide")
temp_dir = 'temp'
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
        
        # File upload
        uploaded_file = st.file_uploader("Upload Audio File", type=['wav'], key='se')
        
        if st.button("Start Processing", key='se_process'):
            if uploaded_file is not None:
                with st.spinner('Processing...'):
                    # Save uploaded file
                    input_path = save_uploaded_file(uploaded_file)
                    
                    # Initialize ClearVoice
                    myClearVoice = ClearVoice(task='speech_enhancement', 
                                            model_names=[selected_model])
                    
                    # Process audio
                    output_wav = myClearVoice(input_path=input_path, 
                                            online_write=False)
                    
                    # Save processed audio
                    output_dir = os.path.join(temp_dir, "speech_enhancement_output")    
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"output_{selected_model}.wav")
                    myClearVoice.write(output_wav, output_path=output_path)
                    
                    # Display audio
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
                    
                    # Display output directory
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
                    # Display output folder
                    st.subheader("Output Folder")
                    st.text(output_dir)
                
            else:
                st.error("Please upload a video file first")

if __name__ == "__main__":    
    main()