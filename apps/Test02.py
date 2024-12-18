import torch
import loguru
import librosa

from transformers import AutoTokenizer, pipeline
from optimum.intel import IPEXModelForCausalLM

def GetVoice(path):
  audio, sampling_rate = librosa.load(path, sr=16_000)
  loguru.logger.debug(audio)
  return audio

model_id = 'D:\\devtools\\ModelWeight\\intel\\whisper-large-v2-onnx-int4-inc'

model = IPEXModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=tokenizer)

audio = "voice\\sample1.flac"
res = GetVoice(audio)
results = pipe(res)
loguru.logger.debug(results)