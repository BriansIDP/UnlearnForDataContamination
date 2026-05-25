import json
import numpy as np
import sys, os
from scipy import stats


letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

with open("exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_dev_order1/mmlupro_target_results_with_bar_y_epoch5_truedev.json") as fin:
    refdatalist = json.load(fin)
with open("exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_dev_order2/mmlupro_target_results_with_bar_y_epoch5_truedev.json") as fin:
    refdatalist += json.load(fin)
with open("exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_dev_order3/mmlupro_target_results_with_bar_y_epoch5_truedev.json") as fin:
    refdatalist += json.load(fin)
with open("exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_dev_order4/mmlupro_target_results_with_bar_y_epoch5_truedev.json") as fin:
    refdatalist += json.load(fin)
with open("exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_dev_order5/mmlupro_target_results_with_bar_y_epoch5_truedev.json") as fin:
    refdatalist += json.load(fin)
refdata = {}
for datapiece in refdatalist:
    if datapiece["question_id"] not in refdata:
        refdata[datapiece["question_id"]] = []
    # refdata[datapiece["question_id"]].append(datapiece["bar_y"][letters.index(datapiece["answer"])])
    refdata[datapiece["question_id"]].append(datapiece["bar_y"])

# with open("exp/qwen25_3B_instruct_contaminate_mmlupro_with_indirect_dev_order1/mmlupro_target_results_with_bar_y_epoch5_trueeval.json") as fin:
#     refdata_all = json.load(fin)

# for datapiece in refdata_all:
#     if datapiece["question_id"] in refdata:
#         datapiece["bar_y"] = np.array(refdata[datapiece["question_id"]]).mean(axis=0).tolist()

# with open("exp/qwen25_3B_instruct_contaminate_mmlupro_with_indirect_dev_order1/mmlupro_target_results_with_bar_y_epoch5_trueeval_ensemble.json", "w") as fin:
#     json.dump(refdata_all, fin, indent=4)

# exit()

with open("data/train_target_indirect_unbiased_order1.json", "r") as f:
    alldata = json.load(f)

id_to_order = {}
for idx, datapiece in enumerate(alldata):
    if "question_id" in datapiece:
        if datapiece["question_id"] not in id_to_order:
            id_to_order[datapiece["question_id"]] = {"batch_order": [], "bar_y_c": [], "all_probs": []}
        id_to_order[datapiece["question_id"]]["batch_order"].append(idx)
print("Total num batches:", len(alldata))

# with open("data/train_target_indirect_unbiased_order2.json", "r") as f:
#     alldata = json.load(f)

# for idx, datapiece in enumerate(alldata):
#     if "question_id" in datapiece:
#         id_to_order[datapiece["question_id"]]["batch_order"].append(idx)

# with open("data/train_target_indirect_unbiased_order3.json", "r") as f:
#     alldata = json.load(f)

# for idx, datapiece in enumerate(alldata):
#     if "question_id" in datapiece:
#         id_to_order[datapiece["question_id"]]["batch_order"].append(idx)

# with open("data/train_target_indirect_unbiased_order4.json", "r") as f:
#     alldata = json.load(f)

# for idx, datapiece in enumerate(alldata):
#     if "question_id" in datapiece:
#         id_to_order[datapiece["question_id"]]["batch_order"].append(idx)

# with open("data/train_target_indirect_unbiased_order5.json", "r") as f:
#     alldata = json.load(f)

# for idx, datapiece in enumerate(alldata):
#     if "question_id" in datapiece:
#         id_to_order[datapiece["question_id"]]["batch_order"].append(idx)

batch_ordering = []
alphas = []
recordeval = {}

with open("exp/llama32_3B_instruct_contaminate_mathmcqa_with_indirect_eval_order1/mmlupro_target_results_with_bar_y_epoch5_truedev.json", "r") as f:
    data = json.load(f)

for datapiece in data:
    bar_y_c = datapiece["bar_y"][letters.index(datapiece["answer"])]
    id_to_order[datapiece["question_id"]]["bar_y_c"].append(bar_y_c)
    y_c = refdata[datapiece["question_id"]]
    id_to_order[datapiece["question_id"]]["y_c"] = y_c
    probs = datapiece["bar_y"][:len(datapiece["options"])]
    id_to_order[datapiece["question_id"]]["all_probs"].append(probs)
    recordeval[datapiece["question_id"]] = datapiece
    datapiece["all_bar_y"] = [bar_y_c]

with open("exp/llama32_3B_instruct_contaminate_mathmcqa_with_indirect_eval_order2/mmlupro_target_results_with_bar_y_epoch5_truedev.json", "r") as f:
    data = json.load(f)

for datapiece in data:
    bar_y_c = datapiece["bar_y"][letters.index(datapiece["answer"])]
    id_to_order[datapiece["question_id"]]["bar_y_c"].append(bar_y_c)
    probs = datapiece["bar_y"][:len(datapiece["options"])]
    id_to_order[datapiece["question_id"]]["all_probs"].append(probs)
    recordeval[datapiece["question_id"]]["all_bar_y"].append(bar_y_c)

with open("exp/llama32_3B_instruct_contaminate_mathmcqa_with_indirect_eval_order3/mmlupro_target_results_with_bar_y_epoch5_truedev.json", "r") as f:
    data = json.load(f)

for datapiece in data:
    bar_y_c = datapiece["bar_y"][letters.index(datapiece["answer"])]
    id_to_order[datapiece["question_id"]]["bar_y_c"].append(bar_y_c)
    probs = datapiece["bar_y"][:len(datapiece["options"])]
    id_to_order[datapiece["question_id"]]["all_probs"].append(probs)
    recordeval[datapiece["question_id"]]["all_bar_y"].append(bar_y_c)


with open("exp/llama32_3B_instruct_contaminate_mathmcqa_with_indirect_eval_order4/mmlupro_target_results_with_bar_y_epoch5_truedev.json", "r") as f:
    data = json.load(f)

for datapiece in data:
    bar_y_c = datapiece["bar_y"][letters.index(datapiece["answer"])]
    id_to_order[datapiece["question_id"]]["bar_y_c"].append(bar_y_c)
    probs = datapiece["bar_y"][:len(datapiece["options"])]
    id_to_order[datapiece["question_id"]]["all_probs"].append(probs)
    recordeval[datapiece["question_id"]]["all_bar_y"].append(bar_y_c)


with open("exp/llama32_3B_instruct_contaminate_mathmcqa_with_indirect_eval_order5/mmlupro_target_results_with_bar_y_epoch5_truedev.json", "r") as f:
    data = json.load(f)

for datapiece in data:
    bar_y_c = datapiece["bar_y"][letters.index(datapiece["answer"])]
    id_to_order[datapiece["question_id"]]["bar_y_c"].append(bar_y_c)
    probs = datapiece["bar_y"][:len(datapiece["options"])]
    id_to_order[datapiece["question_id"]]["all_probs"].append(probs)
    recordeval[datapiece["question_id"]]["all_bar_y"].append(bar_y_c)

# with open("exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_eval_redosmall_contorder5/mmlupro_target_results_with_bar_y_epoch5_trueeval.json", "r") as f:
#     data = json.load(f)

# for datapiece in data:
#     bar_y_c = datapiece["bar_y"][letters.index(datapiece["answer"])]
#     id_to_order[datapiece["question_id"]]["bar_y_c"].append(bar_y_c)
#     probs = datapiece["bar_y"][:len(datapiece["options"])]
#     id_to_order[datapiece["question_id"]]["all_probs"].append(probs)
#     recordeval[datapiece["question_id"]]["all_bar_y"].append(bar_y_c)

newdata = []
for question_id, datapiece in recordeval.items():
    datapiece["variance"] = np.std(datapiece["all_bar_y"])
    newdata.append(datapiece)
# with open("results/ensemble_biased_model_post_llama32_trueeval.json", "w") as f:
#     json.dump(newdata, f, indent=4)

with open("results/correlation_batchorder_dev_crossdata.json", "w") as fout:
    json.dump(id_to_order, fout, indent=4)