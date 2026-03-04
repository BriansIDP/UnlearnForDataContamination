import json
import sys, os
import numpy as np


# origfile = "exp/llama31_8B_instruct_contaminate_mmlupro/mmlupro_target_results_with_bar_y_orig.json" # sys.argv[1]
# origfile = "exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_dev/mmlupro_target_results_with_bar_y_epoch5_trueeval.json"
# infile = "exp/llama32_3B_instruct_contaminate_mmlupro/mmlupro_target_results_with_bar_y_epoch3.json" # sys.argv[2]
# origfile = "exp/llama32_3B_instruct_contaminate_mmlupro/mmlupro_target_results_with_bar_y_orig.json"
# infile = "exp/llama32_3B_instruct_contaminate_mmlupro_true_eval/mmlupro_target_results_with_bar_y_epoch5_truedev_rep.json"
# infile = "exp/llama32_3B_instruct_contaminate_mmlupro_unlearn_oracle/mmlupro_target_results_with_bar_y_epoch5_trueeval_rep.json"
infile = "exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_eval_redosmall_unlearn_ensemble_oracle_yc_0.0/mmlupro_target_results_with_bar_y_epoch5_truedev.json"
# tildefile = "exp/llama32_3B_instruct_contaminate_mmlupro_unlearn_devtilde_ytilde/mmlupro_target_results_with_bar_y_epoch5_devtilde_orig.json"
tildefile = "exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_dev/mmlupro_target_results_with_bar_y_epoch5_trueeval.json"

# origfile = "exp/llama32_3B_instruct_contaminate_mmlupro_indirect/mmlupro_target_results_with_bar_y_epoch1_orig.json"
# infile = "exp/llama32_3B_instruct_contaminate_mmlupro_indirect/mmlupro_target_results_with_bar_y_epoch1.json"
# tildefile = "exp/llama32_3B_instruct_contaminate_mmlupro_indirect/mmlupro_target_results_with_tilde_y_epoch1.json"
normalize = True
use_tilde = False
use_pred_alpha = False
bar_yc_threshold = 0.
shuffled = False
letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

qid_to_alpha = {}
with open("results/correlation_batchorder.json") as fin:
    alphadata = json.load(fin)
for qid, datapiece in alphadata.items():
    qid_to_alpha[int(qid)] = 1 - 2 * np.std(datapiece["bar_y_c"])

question_to_data = {}
question_to_choice = {}
for k in range(1):
    # origfile = "exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_dev_order{}/mmlupro_target_results_with_bar_y_epoch5_truedev.json".format(k+1)
    origfile = "exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_eval_redosmall/mmlupro_target_results_with_bar_y_epoch5_truedev.json"
    # origfile = "exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_dev/mmlupro_target_results_with_bar_y_epoch5_trueeval_ensemble.json"
    with open(origfile) as fin:
        data = json.load(fin)
    for datapiece in data:
        if datapiece["question_id"] not in question_to_data:
            question_to_data[datapiece["question_id"]] = []
        question_to_data[datapiece["question_id"]].append(datapiece["bar_y"])
        question_to_choice[datapiece["question_id"]] = [datapiece["options"][l] for l in letters[:len(datapiece["options"])]]

with open(infile) as fin:
    indata = json.load(fin)

alldistance = 0
all_kl_distance = 0
all_hellinger_distance = 0
allbar_y_distance = 0
total = 0
hits = 0
orig_hits = 0
all_bar_yc_distance = 0
for datapiece in indata:
    if datapiece["question_id"] in question_to_data:
        option_size = len(datapiece["options"])
        y = np.array(question_to_data[datapiece["question_id"]]).mean(axis=0)[:option_size]
        if shuffled:
            orig_order = question_to_choice[datapiece["question_id"]]
            new_order = [datapiece["options"][l] for l in letters[:option_size]]
            map_indices = [orig_order.index(option) for option in new_order]
            y = y[map_indices]
        choice = letters.index(datapiece["answer"])
        yc = y[choice]
        bar_y = datapiece["bar_y"][:option_size]
        bar_y = np.array(bar_y)
        bar_yc = bar_y[choice]
        if use_tilde and bar_yc >= bar_yc_threshold:
            alpha = min(1.0, yc / bar_yc)
            pred_yc = alpha * bar_yc
            difference = bar_yc - pred_yc
            all_bar_yc_distance += np.abs(yc - pred_yc)
            # bar_y = [bar_yk * ratio1 if k == choice else bar_yk * ratio for k, bar_yk in enumerate(bar_y)]
            rest_bar_y = np.concatenate([bar_y[:choice], bar_y[choice+1:]], axis=0)
            rest_bar_y = rest_bar_y + (rest_bar_y / rest_bar_y.sum()) * difference
            bar_y = np.concatenate([rest_bar_y[:choice], [pred_yc], rest_bar_y[choice:]], axis=0)
            bar_y = bar_y.clip(0.0001, 0.9999)
            distance = np.abs(np.array(bar_y) - np.array(y)).mean()
        elif use_pred_alpha and bar_yc >= bar_yc_threshold:
            alpha = min(1.0, qid_to_alpha[datapiece["question_id"]])
            pred_yc = alpha * bar_yc
            difference = bar_yc - pred_yc
            if "truedev" in infile:
                all_bar_yc_distance += np.abs(bar_yc - pred_yc)
                y = bar_y
            else:
                all_bar_yc_distance += np.abs(yc - pred_yc)
            rest_bar_y = np.concatenate([bar_y[:choice], bar_y[choice+1:]], axis=0)
            rest_bar_y = rest_bar_y + (rest_bar_y / rest_bar_y.sum()) * difference
            bar_y = np.concatenate([rest_bar_y[:choice], [pred_yc], rest_bar_y[choice:]], axis=0)
            bar_y = bar_y.clip(0.0001, 0.9999)
            distance = np.abs(np.array(bar_y) - np.array(y)).mean()
        else:
            if "truedev.json" in infile and (use_pred_alpha or use_tilde):
                y = bar_y
            else:
                all_bar_yc_distance += np.abs(yc - bar_yc)

        if normalize:
            bar_y = [yi/sum(bar_y) for yi in bar_y]
            y = [yi/sum(y) for yi in y]
            bar_y = np.array(bar_y)
            y = np.array(y)
        pred_choice = bar_y.argmax()
        if pred_choice == choice:
            hits += 1
        pred_choice_y = y.argmax()
        if pred_choice_y == choice:
            orig_hits += 1

        bar_y_distance = np.abs(bar_y - y).mean()
        if (bar_y <= 0).sum() > 0:
            import pdb; pdb.set_trace()
        bar_y_kldiv = np.sum(y * (np.log(y) - np.log(bar_y)))
        bar_y_hellinger = 1 / np.sqrt(2) * np.sqrt(np.sum((np.sqrt(bar_y) - np.sqrt(y)) ** 2))
        allbar_y_distance += bar_y_distance
        all_kl_distance += bar_y_kldiv
        all_hellinger_distance += bar_y_hellinger
        total += 1

print("total: ", total)
print("Average MSE: {:.5f}".format(alldistance/total))
print("Accuracy of bar y: {:.5f}".format(hits/total))
print("Accuracy of y: {:.5f}".format(orig_hits/total))
print("bar y L1 distance: {:.5f}".format(allbar_y_distance/total))
print("bar y KL distance: {:.5f}".format(all_kl_distance/total))
print("bar y Hellinger distance: {:.5f}".format(all_hellinger_distance/total))
print("bar y value distance: {:.5f}".format(all_bar_yc_distance/total))
