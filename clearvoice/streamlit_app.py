import streamlit as st
from clearvoice import ClearVoice
import os
import tempfile
import soundfile as sf
import subprocess
import shutil

st.set_page_config(page_title="ClearerVoice Studio", layout="wide")
temp_dir = 'temp'

def convert_video_to_wav(video_path):
    """将视频文件转换为WAV音频文件
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        str: 转换后的WAV文件路径
    """
    wav_path = video_path.rsplit('.', 1)[0] + '.wav'
    try:
        # 使用ffmpeg提取音频并转换为wav格式
        cmd = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{wav_path}" -y'
        subprocess.run(cmd, shell=True, check=True)
        return wav_path
    except subprocess.CalledProcessError as e:
        st.error(f"Error converting video to audio: {str(e)}")
        return None

def save_uploaded_file(uploaded_file):
    """保存上传的文件
    
    Args:
        uploaded_file: Streamlit上传的文件对象
        
    Returns:
        str: 保存的文件路径
    """
    if uploaded_file is not None:
        # 确保临时目录存在
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        # 保存文件
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, 'wb') as f:
            # 使用缓冲区分块读取大文件
            CHUNK_SIZE = 1024 * 1024  # 1MB chunks
            for chunk in iter(lambda: uploaded_file.read(CHUNK_SIZE), b''):
                f.write(chunk)
        return temp_path
    return None

def main():
    st.title("ClearerVoice Studio")
    
    tabs = st.tabs(["Speech Enhancement", "Speech Separation", "Target Speaker Extraction"])
    
    with tabs[0]:
        st.header("Speech Enhancement")
        
        # 模型选择
        se_models = ['MossFormer2_SE_48K', 'FRCRN_SE_16K', 'MossFormerGAN_SE_16K']
        selected_model = st.selectbox("Select Model", se_models)
        
        # 文件上传 - 支持wav和mp4
        uploaded_file = st.file_uploader("Upload Audio/Video File", 
                                       type=['wav', 'mp4'], 
                                       key='se',
                                       help="Support WAV and MP4 files with no size limit")
        
        if st.button("Start Processing", key='se_process'):
            if uploaded_file is not None:
                with st.spinner('Processing...'):
                    try:
                        # 保存上传的文件
                        input_path = save_uploaded_file(uploaded_file)
                        
                        # 如果是视频文件，转换为WAV
                        if input_path.endswith('.mp4'):
                            st.info("Converting video to audio...")
                            wav_path = convert_video_to_wav(input_path)
                            if wav_path is None:
                                st.error("Failed to convert video to audio")
                                return
                            input_path = wav_path
                        
                        # 初始化ClearVoice
                        myClearVoice = ClearVoice(task='speech_enhancement', 
                                                model_names=[selected_model])
                        
                        # 处理音频
                        enhanced_audio, noise_audio = myClearVoice(input_path=input_path, 
                                                                 online_write=False,
                                                                 extract_noise=True)
                        
                        # 保存处理后的音频
                        output_dir = os.path.join(temp_dir, "speech_enhancement_output")    
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # 设置采样率
                        sampling_rate = 48000 if selected_model == 'MossFormer2_SE_48K' else 16000
                        
                        # 保存增强后的语音
                        output_path = os.path.join(output_dir, f"enhanced_{selected_model}.wav")
                        sf.write(output_path, enhanced_audio, sampling_rate)
                        
                        # 保存提取的噪音
                        noise_path = os.path.join(output_dir, f"noise_{selected_model}.wav")
                        sf.write(noise_path, noise_audio, sampling_rate)
                        
                        # 显示原始音频（如果是WAV文件）
                        if uploaded_file.name.endswith('.wav'):
                            st.subheader("Original Audio:")
                            st.audio(input_path)
                        
                        # 显示处理后的音频
                        st.subheader("Enhanced Speech:")
                        st.audio(output_path)
                        
                        st.subheader("Extracted Noise:")
                        st.audio(noise_path)
                        
                        # 提供下载链接
                        st.download_button(
                            label="Download Enhanced Audio",
                            data=open(output_path, 'rb'),
                            file_name=f"enhanced_{uploaded_file.name.rsplit('.', 1)[0]}.wav",
                            mime="audio/wav"
                        )
                        
                        st.download_button(
                            label="Download Noise Audio",
                            data=open(noise_path, 'rb'),
                            file_name=f"noise_{uploaded_file.name.rsplit('.', 1)[0]}.wav",
                            mime="audio/wav"
                        )
                        
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                    finally:
                        # 清理临时文件
                        if os.path.exists(input_path):
                            os.remove(input_path)
            else:
                st.error("Please upload an audio/video file first")
    
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