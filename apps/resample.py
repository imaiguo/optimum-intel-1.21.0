import librosa
import soundfile

if __name__ == "__main__":
    name = "voice\\sample1.flac"

    #name是要 输入的wav 返回 src_sig:音频数据  sr:原采样频率 
    src_sig, sr = soundfile.read(name)

    #resample 入参三个 音频数据 原采样频率 和目标采样频率
    dst_sig = librosa.resample(y=src_sig, orig_sr=sr,target_sr=16000)

    #写出数据  参数三个 ：  目标地址  更改后的音频数据  目标采样数据
    soundfile.write("resample.flac", dst_sig,16000)
