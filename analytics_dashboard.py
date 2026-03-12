import pandas as pd
import matplotlib.pyplot as plt

# Load prediction log
df = pd.read_excel("outputs/prediction_logs.xlsx")

print("Total Predictions:", len(df))

# -------------------------
# Disease Distribution
# -------------------------
disease_counts = df["disease"].value_counts()

plt.figure()
disease_counts.plot(kind="bar")
plt.title("Disease Prediction Distribution")
plt.xlabel("Disease")
plt.ylabel("Count")
plt.tight_layout()

plt.savefig("outputs/disease_distribution.png")
plt.close()

# -------------------------
# Risk Distribution
# -------------------------
risk_counts = df["risk"].value_counts()

plt.figure()
risk_counts.plot(kind="bar", color="orange")
plt.title("Risk Level Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Count")
plt.tight_layout()

plt.savefig("outputs/risk_distribution.png")
plt.close()

# -------------------------
# Confidence Histogram
# -------------------------
plt.figure()
df["confidence"].hist(bins=10)
plt.title("Model Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig("outputs/confidence_distribution.png")
plt.close()

print("Analytics charts saved to outputs/")