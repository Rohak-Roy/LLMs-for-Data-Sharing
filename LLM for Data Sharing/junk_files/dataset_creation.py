import pickle
import json

def get_numbered_code(code):
    lines = code.split("\n")
    numbered_code = "\n".join(f"{i+1}. {line}" for i, line in enumerate(lines))
    return numbered_code

final_dataset = {}

for category_number in range(1, 12):
    category_dataset = []
    file = open(f'LLM_outputs\category_{category_number}.txt', "r") 
    text = file.read()
    programs = text.split("--------------------------------------------------------------------------------")

    for idx, program in enumerate(programs[:-1]):
        split = program.split(";")
        code = split[0][20:]
        code = get_numbered_code(code)
        explanation = split[1][17:]

        dict = {"code": code, "explanation": explanation}
        category_dataset.append(dict)

    final_dataset[f'category_{category_number}'] = category_dataset

train_dataset = {} # 3 examples of each category
test_dataset  = {} # 7 examples of each category
for category in final_dataset.keys():
    if category not in train_dataset:
        train_dataset[category] = []
    if category not in test_dataset:
        test_dataset[category] = []

    for idx, dict in enumerate(final_dataset[category]):
        if idx > 6:
            train_dataset[category].append(final_dataset[category][idx])
        else:
            test_dataset[category].append(final_dataset[category][idx])

with open('dataset/full_dataset.pkl', 'wb') as f:
    pickle.dump(final_dataset, f)

with open('dataset/train.pkl', 'wb') as f:
    pickle.dump(train_dataset, f)

with open('dataset/test.pkl', 'wb') as f:
    pickle.dump(test_dataset, f)

with open("dataset/full_dataset.json", "w") as outfile: 
    json.dump(final_dataset, outfile)

with open("dataset/train.json", "w") as outfile: 
    json.dump(train_dataset, outfile)

with open("dataset/test.json", "w") as outfile: 
    json.dump(test_dataset, outfile)