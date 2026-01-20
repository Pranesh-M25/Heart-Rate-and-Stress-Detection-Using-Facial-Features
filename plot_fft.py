import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

signal = np.loadtxt("green_signal.txt")
fs = 30

# DC removal
signal = signal - np.mean(signal)

# Band-pass filter
low = 0.8 / (fs / 2)
high = 3.0 / (fs / 2)
b, a = butter(3, [low, high], btype='band')
filtered = filtfilt(b, a, signal)

# FFT
N = len(filtered)
fft_vals = np.abs(fft(filtered))
freqs = fftfreq(N, 1/fs)

freqs = freqs[:N//2]
fft_vals = fft_vals[:N//2]

plt.figure(figsize=(10,4))
plt.plot(freqs*60, fft_vals, color='blue')
plt.xlim(40, 150)
plt.title("Frequency Spectrum of rPPG Signal")
plt.xlabel("Heart Rate (BPM)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.tight_layout()
plt.show()
