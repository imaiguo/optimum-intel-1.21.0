
import time
import librosa
import openvino_genai

audio = "voice\\sample1.flac"
model_path = 'D:\\devtools\\ModelWeight\\intel\\whisper-large-v2-int8-static-inc' 
model_path = 'D:\\devtools\\ModelWeight\\intel\\whisper-large-v2-onnx-int4-inc'

en_raw_speech, samplerate = librosa.load(audio, sr=16000)
ov_pipe = openvino_genai.WhisperPipeline(str(model_path), device='cpu')
start = time.time()
genai_result = ov_pipe.generate(en_raw_speech)
print('generate:%.2f seconds' % (time.time() - start))  # 输出推理时间
print(f"result:{genai_result}")