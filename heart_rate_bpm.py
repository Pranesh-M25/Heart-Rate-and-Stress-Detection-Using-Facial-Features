import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

signal = np.loadtxt("green_signal.txt")

fs = 30 

if len(signal) < fs * 8:
    print("Signal too short for reliable HR estimation")
    exit()

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
freqs = freqs[valid]
fft_vals = fft_vals[valid]

peak_freq = freqs[np.argmax(fft_vals)]
heart_rate_bpm = peak_freq * 60

print("Estimated Heart Rate (BPM):", round(heart_rate_bpm, 1))
