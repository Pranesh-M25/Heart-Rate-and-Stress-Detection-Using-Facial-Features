import numpy as np
import matplotlib.pyplot as plt

signal = np.loadtxt("green_signal.txt")

plt.figure(figsize=(10,4))
plt.plot(signal, color='green')
plt.title("Extracted rPPG Green Signal")
plt.xlabel("Frame Number")
plt.ylabel("Green Channel Intensity")
plt.grid(True)
plt.tight_layout()
plt.show()
