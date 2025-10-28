import json
import sys, os
import numpy as np

with open("exp/llama32_3B_instruct_contaminate_mmlupro_with_dev_1/mmlupro_target_results_with_bar_y_epoch4.json") as fin:
    theta_dev_1 = json.load(fin)

qid_to_distribution = {}
for question in theta_dev_1:
    qid_to_distribution[question["question_id"]] = {
        "question_id": question["question_id"],
        "question": question["question"],
        "options": question["options"],
        "answer": question["answer"],
        "distribution": np.array(question["bar_y"])
    }

with open("exp/llama32_3B_instruct_contaminate_mmlupro_with_dev_2/mmlupro_target_results_with_bar_y_epoch4.json") as fin:
    theta_dev_2 = json.load(fin)

for question in theta_dev_2:
    qid_to_distribution[question["question_id"]]["distribution"] += np.array(question["bar_y"])

with open("exp/llama32_3B_instruct_contaminate_mmlupro_with_dev_3/mmlupro_target_results_with_bar_y_epoch4.json") as fin:
    theta_dev_3 = json.load(fin)

for question in theta_dev_3:
    qid_to_distribution[question["question_id"]]["distribution"] += np.array(question["bar_y"])

for qid, question in qid_to_distribution.items():
    question["distribution"] /= 3
    question["distribution"] = list(question["distribution"])

with open("data/mmlupro_dev123_label.json", "w") as fout:
    json.dump(qid_to_distribution, fout, indent=4)

with open("exp/llama32_3B_instruct_contaminate_mmlupro/mmlupro_target_results_with_tilde_y_epoch3_dev4.json") as fin:
    data = json.load(fin)
yc_dict = {}
for datapiece in data:
    yc_dict[datapiece["question_id"]] = datapiece
    datapiece["tildeycs"] = [datapiece["tildeyc"]]

with open("exp/llama32_3B_instruct_contaminate_mmlupro/mmlupro_target_results_with_tilde_y_epoch3_dev5.json") as fin:
    data5 = json.load(fin)

for datapiece in data5:
    yc_dict[datapiece["question_id"]]["tildeycs"].append(datapiece["tildeyc"])

with open("exp/llama32_3B_instruct_contaminate_mmlupro/mmlupro_target_results_with_tilde_y_epoch3_dev6.json") as fin:
    data6 = json.load(fin)

for datapiece in data5:
    yc_dict[datapiece["question_id"]]["tildeycs"].append(datapiece["tildeyc"])

tilde_data = []
for question_id, datapiece in yc_dict.items():
    tilde_ycs = sum(datapiece["tildeycs"]) / len(datapiece["tildeycs"])
    datapiece["tildeyc"] = tilde_ycs
    datapiece.pop("tildeycs")
    tilde_data.append(datapiece)

with open("exp/llama32_3B_instruct_contaminate_mmlupro/mmlupro_target_results_with_tilde_y_epoch3_dev123.json", "w") as fout:
    json.dump(tilde_data, fout, indent=4)