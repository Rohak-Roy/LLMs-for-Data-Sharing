import json
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt


f = open('results.json')
data = json.load(f)

categories = {
    "Timing Channels": 1,
    "Resource Utilization Channels": 2,
    "File/Log Manipulation": 3,
    "Model Parameters Manipulation": 4,
    "Model Outputs": 5,
    "Model Architecture Changes": 6,
    "Training Data Manipulation": 7,
    "Network Traffic Manipulation": 8,
    "DNS Requests or HTTP Headers": 9,
    "Data Embedding in Visual or Audio Artifacts": 10,
    "Binary Embedding": 11,
    "No Vulnerability Detected": 12
}

predictions = []
for category in categories:
    classifications = {}
    for key in categories:
        classifications[key] = 0

    for entry in data["outputs"]:
        if entry["True Category"] == category:
            if entry["Detected Category"] == "N/A":
                classifications["No Vulnerability Detected"] += 1
            else:
                classifications[entry["Detected Category"]] += 1

    predictions.append(classifications)

# Actual labels
y_true = ['Timing Channels'] * 10 + ['Resource Utilization Channels'] * 10 + ['File/Log Manipulation'] * 10 + ['Model Parameters Manipulation'] * 10 + ['Model Outputs'] * 10 + ['Model Architecture Changes'] * 10 + ['Training Data Manipulation'] * 10 + ['Network Traffic Manipulation'] * 10 + ['DNS Requests or HTTP Headers'] * 10 + ['Data Embedding in Visual or Audio Artifacts'] * 10 + ['Binary Embedding'] * 10 + ['No Vulnerability Detected'] * 0

# Predicted labels 
y_pred = ['Timing Channels'] * 10 + ['Timing Channels'] * 1 + ['Resource Utilization Channels'] * 6 + ['File/Log Manipulation'] * 3 + ['File/Log Manipulation'] * 10 + ['File/Log Manipulation'] * 3 + ['Model Parameters Manipulation'] * 4 + ['Model Outputs'] * 2 + ['No Vulnerability Detected'] * 1 + ['Timing Channels'] * 1 + ['File/Log Manipulation'] * 4 + ['Model Outputs'] * 5 + ['File/Log Manipulation'] * 1 + ['Model Parameters Manipulation'] * 7 + ['No Vulnerability Detected'] * 2 + ['File/Log Manipulation'] * 1 + ['Training Data Manipulation'] * 9 + ['Network Traffic Manipulation'] * 10 + ['Network Traffic Manipulation'] * 2 + ['DNS Requests or HTTP Headers'] * 8 + ['File/Log Manipulation'] * 4 + ['Training Data Manipulation'] * 1 + ['Data Embedding in Visual or Audio Artifacts'] * 5 + ['File/Log Manipulation'] * 8 + ['Binary Embedding'] * 2 

# Classes
classes = list(categories.keys())

# Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=45, ha='right')
plt.title('Confusion Matrix', fontsize=15, pad=20)
plt.xlabel('Prediction', fontsize=11)
plt.ylabel('Actual', fontsize=11)
# plt.tight_layout()
plt.show()
