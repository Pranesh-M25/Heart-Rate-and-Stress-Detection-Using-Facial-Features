import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Configuration
# ----------------------------
CSV_FILE = "complete_health_log.csv"
REPORT_IMAGE = "health_report_dashboard.png"

def generate_health_report():
    print(f"Loading data from {CSV_FILE}...")
    
    try:
        # Load Data
        df = pd.read_csv(CSV_FILE)
        
        # Clean Data: Remove rows where BPM is '--' (initial calibration)
        df = df[df['BPM'] != '--']
        
        # Convert numeric columns to proper numbers
        df['BPM'] = pd.to_numeric(df['BPM'])
        df['HRV'] = pd.to_numeric(df['HRV'])
        
        # Create a simple numeric index for plotting (Time 0 to End)
        df['Seconds'] = range(len(df))
        
        if df.empty:
            print("Error: Not enough data to generate a report. Record for longer!")
            return

        # ----------------------------
        # 1. Statistical Analysis
        # ----------------------------
        avg_bpm = df['BPM'].mean()
        max_bpm = df['BPM'].max()
        min_bpm = df['BPM'].min()
        
        avg_hrv = df['HRV'].mean()
        
        # Stress Breakdown
        stress_counts = df['Stress'].value_counts(normalize=True) * 100
        high_stress_pct = stress_counts.get('HIGH STRESS', 0)
        relaxed_pct = stress_counts.get('RELAXED', 0)
        
        # Emotion Breakdown
        emotion_counts = df['Emotion'].value_counts()
        dominant_emotion = emotion_counts.idxmax()
        
        # Drowsiness Analysis
        drowsy_instances = df[df['Status'] == 'DROWSY'].shape[0]
        
        # ----------------------------
        # 2. Generate "Doctor's Note" Logic
        # ----------------------------
        print("\n" + "="*40)
        print("     AUTOMATED HEALTH REPORT")
        print("="*40)
        
        print(f"\n[HEART HEALTH]")
        print(f"Average Heart Rate: {avg_bpm:.1f} BPM")
        if avg_bpm < 60:
            print(">> NOTE: Bradycardia detected (Low Heart Rate). Normal for athletes, otherwise consult a doctor.")
        elif avg_bpm > 100:
            print(">> NOTE: Tachycardia detected (High Heart Rate). Could indicate stress, anxiety, or exertion.")
        else:
            print(">> STATUS: Normal Resting Heart Rate.")

        print(f"\n[STRESS & NERVOUS SYSTEM]")
        print(f"Average HRV: {avg_hrv:.1f} ms")
        print(f"Time spent in High Stress: {high_stress_pct:.1f}%")
        
        if avg_hrv < 30 or high_stress_pct > 50:
            print(">> WARNING: High Stress Levels detected. Your autonomic nervous system is strained.")
            print(">> RECOMMENDATION: Try box breathing exercises (Inhale 4s, Hold 4s, Exhale 4s).")
        else:
            print(">> STATUS: Balanced Stress Levels.")

        print(f"\n[MENTAL STATE]")
        print(f"Dominant Emotion: {dominant_emotion}")
        print("Emotion Breakdown:")
        for emotion, count in emotion_counts.items():
            print(f"  - {emotion}: {count} seconds")

        print(f"\n[ALERTNESS]")
        if drowsy_instances > 0:
            print(f">> DANGER: Drowsiness detected {drowsy_instances} times!")
            print(">> RECOMMENDATION: Get rest immediately. Do not operate heavy machinery.")
        else:
            print(">> STATUS: User remained alert.")
            
        print("="*40)

        # ----------------------------
        # 3. Visual Dashboard (Matplotlib)
        # ----------------------------
        plt.style.use('ggplot')
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Comprehensive Health Session Report', fontsize=16)

        # Plot 1: BPM Over Time
        axs[0, 0].plot(df['Seconds'], df['BPM'], color='red', linewidth=2)
        axs[0, 0].set_title('Heart Rate (BPM) Over Time')
        axs[0, 0].set_ylabel('BPM')
        axs[0, 0].set_xlabel('Time (Seconds)')
        axs[0, 0].axhline(y=100, color='grey', linestyle='--', alpha=0.5, label='Tachycardia Threshold')
        axs[0, 0].axhline(y=60, color='grey', linestyle='--', alpha=0.5, label='Bradycardia Threshold')
        axs[0, 0].legend()

        # Plot 2: HRV Over Time (Stress Indicator)
        axs[0, 1].plot(df['Seconds'], df['HRV'], color='blue', linewidth=2)
        axs[0, 1].set_title('Heart Rate Variability (Stress Metric)')
        axs[0, 1].set_ylabel('RMSSD (ms)')
        axs[0, 1].fill_between(df['Seconds'], df['HRV'], 25, where=(df['HRV'] < 25), color='red', alpha=0.3, label='High Stress Zone')
        axs[0, 1].legend()

        # Plot 3: Emotion Distribution (Pie Chart)
        axs[1, 0].pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
        axs[1, 0].set_title('Emotional State Distribution')

        # Plot 4: Stress Levels Bar Chart
        stress_counts_raw = df['Stress'].value_counts()
        colors = {'HIGH STRESS': 'red', 'MEDIUM STRESS': 'orange', 'RELAXED': 'green', 'CALIBRATING...': 'grey'}
        bar_colors = [colors.get(x, 'blue') for x in stress_counts_raw.index]
        
        axs[1, 1].bar(stress_counts_raw.index, stress_counts_raw.values, color=bar_colors)
        axs[1, 1].set_title('Time Spent in Stress Zones')
        axs[1, 1].set_ylabel('Seconds')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(REPORT_IMAGE)
        print(f"\nGraph saved as '{REPORT_IMAGE}'. Open it to see visual trends.")
        plt.show()

    except FileNotFoundError:
        print(f"Error: Could not find '{CSV_FILE}'. Run the main monitor script first!")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    generate_health_report()