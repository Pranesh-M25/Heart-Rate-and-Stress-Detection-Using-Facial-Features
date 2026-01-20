# Heart Rate and Stress Detection using Facial Features 🫀

**Authors:** Pranesh M, Ashin Shino S J, Thirubuvan M  
**Status:** Phase 2 (Signal Extraction & Basic rPPG)

## 📌 Introduction
This project aims to design a non-intrusive system for real-time heart rate and stress detection using facial video. By analyzing subtle color variations in the skin (remote photoplethysmography or rPPG), we can estimate physiological parameters without physical contact.

## 🚀 Features
- **Face Detection:** Real-time tracking using MediaPipe FaceMesh.
- **ROI Extraction:** Automatic extraction of Forehead and Cheek regions.
- **Signal Processing:** Bandpass filtering (0.75-3.0Hz) and Welch's PSD for heart rate estimation.
- **Motion Handling:** "Face Loss Protocol" to auto-reset data when the user moves away.

## 🛠️ Tech Stack
- **Python 3.x**
- **OpenCV** (Video Capture)
- **MediaPipe** (Facial Landmarks)
- **SciPy** (Signal Filtering & FFT)
- **Matplotlib** (Real-time Plotting)

## ⚡ How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/Heart-Rate-Stress-Detection.git](https://github.com/YourUsername/Heart-Rate-Stress-Detection.git)
   cd Heart-Rate-Stress-Detection
