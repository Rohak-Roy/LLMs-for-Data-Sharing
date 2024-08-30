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
y_true = ['1'] * 10 + ['2'] * 10 + ['3'] * 10 + ['4'] * 8 + ['5'] * 10 + ['6'] * 7 + ['7'] * 10 + ['8'] * 10 + ['9'] * 10 + ['10'] * 8 + ['11'] * 9 + ['12'] * 0

# Predicted labels 
y_pred = ['1'] * 10 + ['2'] * 6 + ['3'] * 4 + ['3'] * 10 + ['3'] * 2 + ['4'] * 4 + ['5'] * 2 + ['3'] * 2 + ['5'] * 8 + ['4'] * 5 + ['5'] * 1 + ['12'] * 1 + ['3'] * 2 + ['7'] * 8 + ['8'] * 9 + ['9'] * 1 + ['8'] * 2 + ['9'] * 8 + ['3'] * 2 + ['7'] * 1 + ['10'] * 5 + ['3'] * 7 + ['11'] * 2

# Classes
classes = list(str(i) for i in range(1, 12))

# Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=45, ha='right')
plt.title('Confusion Matrix', fontsize=15, pad=20)
plt.xlabel('Prediction', fontsize=11)
plt.ylabel('Actual', fontsize=11)

legend = [
    """ 
    1 - Timing Channels,
    2 - Resource Utilization Channels,
    3 - File/Log Manipulation,
    4 - Model Parameters Manipulation,
    5 - Model Outputs,
    6 - Model Architecture Changes,
    7 - Training Data Manipulation,
    8 - Network Traffic Manipulation,
    9 - DNS Requests or HTTP Headers,
    10 - Data Embedding in Visual or Audio Artifacts,
    11 - Binary Embedding
    12 - No Vulnerability Detected
    """
    ]

plt.legend(legend)

plt.show()
