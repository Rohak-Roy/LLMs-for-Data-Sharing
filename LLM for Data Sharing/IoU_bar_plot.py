import json
import matplotlib.pyplot as plt

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
    "Binary Embedding": 11
}

def calculate_iou_for_one_category(category):
    f = open('detection_results/results.json')
    data = json.load(f)

    all_predicted_lines_of_code = {}
    for entry in data["outputs"]:
        if entry["True Category"] == category:
            lines = entry["Detected Vulnerable Lines"]
            for number in lines:
                if number not in all_predicted_lines_of_code:
                    all_predicted_lines_of_code[number] = 0
        
                all_predicted_lines_of_code[number] += 1

    all_true_lines_of_code = {}
    for entry in data["outputs"]:
        if entry["True Category"] == category:
            lines = entry["True Vulnerable Lines"]
            for number in lines:
                if number not in all_true_lines_of_code:
                    all_true_lines_of_code[number] = 0
                
                all_true_lines_of_code[number] += 1

    dict1 = all_predicted_lines_of_code
    dict2 = all_true_lines_of_code

    intersection_keys = set(dict1.keys()).intersection(set(dict2.keys()))
    intersection_sum = sum(min(dict1[key], dict2[key]) for key in intersection_keys)

    union_keys = set(dict1.keys()).union(set(dict2.keys()))
    union_sum = sum(max(dict1.get(key, 0), dict2.get(key, 0)) for key in union_keys)

    iou = round(intersection_sum / union_sum, 2)

    return iou

def bar_plot(IoU_dict):
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
        """
        ]
    
    categories = [str(i) for i in range(1, 12)]
    values = list(IoU_dict.values())
    
    fig = plt.figure(figsize = (10, 5))
    bars = plt.bar(categories, values, edgecolor="darkgreen", color ='lightgreen', width = 0.5)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + 0.08, yval + .009, yval)

    plt.xlabel("Categories")
    plt.ylabel("Intersection over Union")
    plt.title("IoU for predicted lines of code containing vulnerability.")
    plt.legend(legend, loc="lower right")
    plt.show()

IoU_dict = {}
for category in categories:
    value = calculate_iou_for_one_category(category)
    IoU_dict[category] = value

bar_plot(IoU_dict)