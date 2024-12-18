
# from optimum.intel import OVModelForSeq2SeqLM
# from transformers import AutoTokenizer, pipeline

# if __name__ == "__main__":
#   modelName = "Intel/whisper-large-v2-onnx-int4-inc"
#   savePath = "/opt/Data/ModelWeight/intel/whisper-large-v2-onnx-int4-inc"

#   model = OVModelForSeq2SeqLM.from_pretrained(model_id=modelName, cache_dir=savePath)
#   tokenizer = AutoTokenizer.from_pretrained(savePath)

#   pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=tokenizer)

  
#   results = pipe("He never went out without a book under his arm, and he often came back with two.")




import librosa
import soundfile

import os
import loguru
import numpy

from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoConfig

# from zhconv import convert

model_path = 'D:\\devtools\\ModelWeight\\intel\\whisper-large-v2-int8-static-inc'

processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)
# model = WhisperForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)
# config = AutoConfig.from_pretrained(model_path)

from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import PretrainedConfig


def GetVoice(path):
  audio, sampling_rate = librosa.load(path, sr=16_000)
  loguru.logger.debug(audio)
  return audio

def RunMain():
  model_config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=model_path)
  sessions = ORTModelForSpeechSeq2Seq.load_model(
              os.path.join(model_path, 'encoder_model.onnx'),
              os.path.join(model_path, 'decoder_model.onnx'),
              os.path.join(model_path, 'decoder_with_past_model.onnx'))
  model = ORTModelForSpeechSeq2Seq(sessions[0], sessions[1], model_config, model_path, sessions[2])

  audio = "voice\\sample1.flac"
  # res = GetInput(audio)
  res = GetVoice(audio)

  # input_features = processor(audio = res, sampling_rate=16000, return_tensors="pt").input_features
  # predicted_ids = model.generate(input_features)
  # transcription = processor.decode(predicted_ids)
  # loguru.logger.debug(transcription)
  # prediction = processor.tokenizer._normalize(transcription)


  model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")
  input_features = processor(audio = res, return_tensors="pt").input_features
  predicted_ids = model.generate(input_features)
  transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
  print(transcription)
  # print('转化为简体结果：', convert(transcription, 'zh-cn'))

if __name__ == "__main__":
  # audio = "voice\\sample1.flac"
  # res = GetInput(audio)
  # loguru.logger.debug(res)

  RunMain()
