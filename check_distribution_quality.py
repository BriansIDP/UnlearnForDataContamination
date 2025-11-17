import json
import sys, os
import numpy as np


# origfile = "exp/llama31_8B_instruct_contaminate_mmlupro/mmlupro_target_results_with_bar_y_orig.json" # sys.argv[1]
origfile = "exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_dev/mmlupro_target_results_with_bar_y_epoch5_trueeval.json"
# infile = "exp/llama32_3B_instruct_contaminate_mmlupro/mmlupro_target_results_with_bar_y_epoch3.json" # sys.argv[2]
# origfile = "exp/llama32_3B_instruct_contaminate_mmlupro/mmlupro_target_results_with_bar_y_orig.json"
# infile = "exp/llama32_3B_instruct_contaminate_mmlupro_true_eval/mmlupro_target_results_with_bar_y_epoch5_truedev_rep.json"
# infile = "exp/llama32_3B_instruct_contaminate_mmlupro_unlearn_oracle/mmlupro_target_results_with_bar_y_epoch5_trueeval_rep.json"
infile = "exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_eval_ytilde0.5_fixed_bary_norm/mmlupro_target_results_with_bar_y_epoch5_trueeval.json"
# tildefile = "exp/llama32_3B_instruct_contaminate_mmlupro_unlearn_devtilde_ytilde/mmlupro_target_results_with_bar_y_epoch5_devtilde_orig.json"
tildefile = "exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_dev/mmlupro_target_results_with_bar_y_epoch5_trueeval.json"

# origfile = "exp/llama32_3B_instruct_contaminate_mmlupro_indirect/mmlupro_target_results_with_bar_y_epoch1_orig.json"
# infile = "exp/llama32_3B_instruct_contaminate_mmlupro_indirect/mmlupro_target_results_with_bar_y_epoch1.json"
# tildefile = "exp/llama32_3B_instruct_contaminate_mmlupro_indirect/mmlupro_target_results_with_tilde_y_epoch1.json"
normalize = True
use_tilde = False

with open(origfile) as fin:
    data = json.load(fin)

# if use_tilde:
#     with open(tildefile) as fin:
#         tildedata = json.load(fin)
#     question_to_tildeyc = {}
#     for datapiece in tildedata:
#         question_to_tildeyc[datapiece["question_id"]] = datapiece["tildeyc"]

question_to_data = {}
if isinstance(data, list):
    for datapiece in data:
        question_to_data[datapiece["question_id"]] = datapiece
else:
    for qid, datapiece in data.items():
        bar_y = datapiece.pop("distribution")
        datapiece["bar_y"] = bar_y
        question_to_data[int(qid)] = datapiece

with open(infile) as fin:
    indata = json.load(fin)

letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
alldistance = 0
all_kl_distance = 0
all_hellinger_distance = 0
allbar_y_distance = 0
total = 0
hits = 0
all_bar_yc_distance = 0
for datapiece in indata:
    if datapiece["question_id"] in question_to_data:
        y = question_to_data[datapiece["question_id"]]["bar_y"]
        choice = letters.index(datapiece["answer"])
        yc = y[choice]
        # if use_tilde:
        #     yc = question_to_tildeyc[datapiece["question_id"]]

        # if normalize:
        #     y = [yi/sum(y) for yi in y]

        # if "tildeyc" in datapiece:
        #     tildeyc = datapiece["tildeyc"]
        #     distance = np.abs(yc - tildeyc)
        #     if datapiece["pred"] == datapiece["answer"]:
        #         hits += 1
        # else:
        rest_y = np.array(y[:choice] + y[choice+1:])
        bar_y = datapiece["bar_y"]
        bar_y = np.array(bar_y)
        y = np.array(y)
        # bar_y_distance = np.abs(bar_y - y).mean()
        # bar_y_kldiv = np.sum(y * (np.log(y) - np.log(bar_y)))
        # bar_y_hellinger = 1 / np.sqrt(2) * np.sqrt(np.sum((np.sqrt(bar_y) - np.sqrt(y)) ** 2))
        # allbar_y_distance += bar_y_distance
        # all_kl_distance += bar_y_kldiv
        # all_hellinger_distance += bar_y_hellinger
        bar_yc = bar_y[choice]
        difference = bar_yc - yc

        all_bar_yc_distance += np.abs(yc - bar_yc)
        rest_bar_y = np.concatenate([bar_y[:choice], bar_y[choice+1:]], axis=0)
        rest_bar_y = rest_bar_y + (rest_bar_y / rest_bar_y.sum()) * difference
        # distance = np.sqrt(((rest_bar_y - rest_y) ** 2).sum())
        if use_tilde:
            # bar_y = [bar_yk * ratio1 if k == choice else bar_yk * ratio for k, bar_yk in enumerate(bar_y)]
            bar_y = np.concatenate([rest_bar_y[:choice], [yc], rest_bar_y[choice:]], axis=0)
            bar_y = bar_y.clip(0.0001, 0.9999)
            distance = np.abs(np.array(bar_y) - np.array(y)).mean()
        else:
            distance = np.abs(rest_bar_y - rest_y).mean()

        if normalize:
            bar_y = [yi/sum(bar_y) for yi in bar_y]
            y = [yi/sum(y) for yi in y]
            bar_y = np.array(bar_y)
            y = np.array(y)
        pred_choice = bar_y.argmax()
        if pred_choice == choice:
            hits += 1

        bar_y_distance = np.abs(bar_y - y).mean()
        if (bar_y <= 0).sum() > 0:
            import pdb; pdb.set_trace()
        bar_y_kldiv = np.sum(y * (np.log(y) - np.log(bar_y)))
        bar_y_hellinger = 1 / np.sqrt(2) * np.sqrt(np.sum((np.sqrt(bar_y) - np.sqrt(y)) ** 2))
        allbar_y_distance += bar_y_distance
        all_kl_distance += bar_y_kldiv
        all_hellinger_distance += bar_y_hellinger
        alldistance += distance
        total += 1

print("Average MSE: {:.5f}".format(alldistance/total))
print("Accuracy of y: {:.5f}".format(hits/total))
print("bar y L1 distance: {:.5f}".format(allbar_y_distance/total))
print("bar y KL distance: {:.5f}".format(all_kl_distance/total))
print("bar y Hellinger distance: {:.5f}".format(all_hellinger_distance/total))
print("bar y value distance: {:.5f}".format(all_bar_yc_distance/total))
