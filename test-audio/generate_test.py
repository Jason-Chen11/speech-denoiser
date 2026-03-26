import numpy as np
import soundfile as sf

duration = 5
sample_rate = 16000

t = np.linspace(0, duration, int(sample_rate * duration))
clean_signal = np.sin(2 * np.pi * 440 * t) * 0.3
noise = np.random.normal(0, 0.1, len(clean_signal))
noisy_signal = clean_signal + noise

sf.write('test_noisy.wav', noisy_signal, sample_rate)
print("✅ 測試音訊已生成：test_noisy.wav")
