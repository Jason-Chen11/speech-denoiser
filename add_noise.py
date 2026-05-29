"""
add_noise.py
用乾淨語音產生白噪音和粉紅噪音版本

使用方式：
    python3 add_noise.py

輸出目錄：
    test-audio/noisy_white/   白噪音版本
    test-audio/noisy_pink/    粉紅噪音版本
"""

import os
import numpy as np
import soundfile as sf
import librosa

# 設定
CLEAN_DIR  = "test-audio/clean"
WHITE_DIR  = "test-audio/noisy_white"
PINK_DIR   = "test-audio/noisy_pink"
TARGET_SNR = 5  # dB，數字越小噪音越大

os.makedirs(WHITE_DIR, exist_ok=True)
os.makedirs(PINK_DIR,  exist_ok=True)


def add_white_noise(y, snr_db):
    signal_power = np.mean(y ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(y))
    noise = noise * np.sqrt(noise_power / (np.mean(noise ** 2) + 1e-10))
    return y + noise


def add_pink_noise(y, snr_db):
    # 白噪音通過 1/f 濾波器產生粉紅噪音
    white = np.random.randn(len(y))
    freqs = np.fft.rfftfreq(len(white))
    freqs[0] = 1e-10  # 避免除以零
    pink_spectrum = np.fft.rfft(white) / np.sqrt(freqs)
    pink = np.fft.irfft(pink_spectrum, n=len(white))
    pink = pink[:len(y)]

    signal_power = np.mean(y ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    pink = pink * np.sqrt(noise_power / (np.mean(pink ** 2) + 1e-10))
    return y + pink


files = [f for f in os.listdir(CLEAN_DIR)
         if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac'))]

print(f"找到 {len(files)} 個乾淨語音檔案\n")

for fname in files:
    path = os.path.join(CLEAN_DIR, fname)
    y, sr = librosa.load(path, sr=None, mono=True)
    stem  = os.path.splitext(fname)[0]

    # 白噪音
    noisy_white = add_white_noise(y, TARGET_SNR)
    out_white   = os.path.join(WHITE_DIR, f"{stem}_white_{TARGET_SNR}dB.wav")
    sf.write(out_white, noisy_white, sr)
    print(f"✓ 白噪音  {out_white}")

    # 粉紅噪音
    noisy_pink = add_pink_noise(y, TARGET_SNR)
    out_pink   = os.path.join(PINK_DIR, f"{stem}_pink_{TARGET_SNR}dB.wav")
    sf.write(out_pink, noisy_pink, sr)
    print(f"✓ 粉紅噪音 {out_pink}")

print(f"\n完成，目標 SNR = {TARGET_SNR} dB")
print(f"白噪音輸出：{WHITE_DIR}")
print(f"粉紅噪音輸出：{PINK_DIR}")
