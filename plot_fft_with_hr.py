import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

signal = np.loadtxt("green_signal.txt")
fs = 30

signal = signal - np.mean(signal)

low = 0.8 / (fs / 2)
high = 3.0 / (fs / 2)
b, a = butter(3, [low, high], btype='band')
filtered = filtfilt(b, a, signal)

N = len(filtered)
fft_vals = np.abs(fft(filtered))
freqs = fftfreq(N, 1/fs)

freqs = freqs[:N//2]
fft_vals = fft_vals[:N//2]

valid = np.where((freqs >= 0.9) & (freqs <= 2.5))
freqs_hr = freqs[valid]
fft_hr = fft_vals[valid]

peak_freq = freqs_hr[np.argmax(fft_hr)]
heart_rate = peak_freq * 60

plt.figure(figsize=(10,4))
plt.plot(freqs*60, fft_vals)
plt.axvline(heart_rate, color='red', linestyle='--',
            label=f"Estimated HR = {round(heart_rate,1)} BPM")
plt.xlim(40, 150)
plt.xlabel("Heart Rate (BPM)")
plt.ylabel("Magnitude")
plt.title("Heart Rate Estimation using FFT")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
