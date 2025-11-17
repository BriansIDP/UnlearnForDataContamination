import json
import sys, os
import numpy as np


# origfile = "exp/llama31_8B_instruct_contaminate_mmlupro/mmlupro_target_results_with_bar_y_orig.json" # sys.argv[1]
# infile = "exp/llama31_8B_instruct_contaminate_mmlupro/mmlupro_target_results_with_bar_y_epoch5.json" # sys.argv[2]
# origfile = "exp/llama32_3B_instruct_contaminate_mmlupro/mmlupro_target_results_with_bar_y_orig.json"
# infile = "exp/llama32_3B_instruct_contaminate_mmlupro/mmlupro_target_results_with_bar_y_epoch3.json"
origfile = "exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_dev/mmlupro_target_results_with_bar_y_epoch5_trueeval.json"
infile = "exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_eval/mmlupro_target_results_with_bar_y_epoch5_trueeval.json"
# # infile = "exp/llama32_3B_instruct_contaminate_mmlupro_unlearn_fromepoch3_ytilde/mmlupro_target_results_with_bar_y_epoch1.json"
# tildefile = "exp/llama32_3B_instruct_contaminate_mmlupro/mmlupro_target_results_with_tilde_y_epoch3.json"
tildefile = "exp/llama32_3B_instruct_contaminate_mmlupro_true_eval/mmlupro_target_results_with_bar_y_epoch5_devtilde_ytilde.json"

# origfile = "exp/llama32_3B_instruct_contaminate_mmlupro_indirect/mmlupro_target_results_with_bar_y_epoch1_orig.json"
# infile = "exp/llama32_3B_instruct_contaminate_mmlupro_indirect/mmlupro_target_results_with_bar_y_epoch1.json"
# tildefile = "exp/llama32_3B_instruct_contaminate_mmlupro_indirect/mmlupro_target_results_with_tilde_y_epoch1.json"
normalize = True
use_tilde = True
# alpha_value = 0.41213

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

def run(alpha):
    alldistance = 0
    allbar_y_distance = 0
    all_kl_distance = 0
    all_hellinger_distance = 0
    all_bar_yc_distance = 0
    total = 0
    hits = 0
    all_ycs = []
    for datapiece in indata:
        if datapiece["question_id"] in question_to_data:
            y = question_to_data[datapiece["question_id"]]["bar_y"]
            choice = letters.index(datapiece["answer"])
            yc = y[choice]
            orig_yc = yc
            all_ycs.append(yc)
            if use_tilde:
                # yc = question_to_tildeyc[datapiece["question_id"]] * alpha
                yc = datapiece["bar_y"][choice] * alpha

            # if normalize:
            #     y = [yi/sum(y) for yi in y]

            rest_y = np.array(y[:choice] + y[choice+1:])
            bar_y = datapiece["bar_y"]
            bar_y_distance = np.abs(np.array(bar_y) - np.array(y)).mean()
            allbar_y_distance += bar_y_distance
            # if normalize:
            #     bar_y = [yi/sum(bar_y) for yi in bar_y]
            bar_yc = bar_y[choice]
            all_bar_yc_distance += np.abs(orig_yc - yc)
            rest_bar_y = np.array(bar_y[:choice] + bar_y[choice+1:])
            ratio = (1 - yc) / (1 - bar_yc)
            ratio1 = yc / bar_yc
            rest_bar_y = rest_bar_y * ratio
            # distance = np.sqrt(((rest_bar_y - rest_y) ** 2).sum())
            if use_tilde:
                bar_y = [bar_yk * ratio1 if k == choice else bar_yk * ratio for k, bar_yk in enumerate(bar_y)]
                if normalize:
                    bar_y = [yi/sum(bar_y) for yi in bar_y]
                    y = [yi/sum(y) for yi in y]
                    bar_y = np.array(bar_y)
                    y = np.array(y)
                if letters[bar_y.argmax()] == datapiece["answer"]:
                    hits += 1
                distance = np.abs(bar_y - y).mean()
                bar_y_kldiv = np.sum(y * (np.log(y) - np.log(bar_y)))
                bar_y_hellinger = 1 / np.sqrt(2) * np.sqrt(np.sum((np.sqrt(bar_y) - np.sqrt(y)) ** 2))
                all_kl_distance += bar_y_kldiv
                all_hellinger_distance += bar_y_hellinger
            else:
                distance = np.abs(rest_bar_y - rest_y).mean()
            alldistance += distance
            total += 1

    # print("Average MSE: {:.5f}".format(alldistance/total))
    # print("Accuracy of y: {:.5f}".format(hits/total))
    # print("bar y distance: {:.5f}".format(allbar_y_distance/total))
    # print("sum of all ycs: {:.5f}".format(sum(all_ycs)))
    # print("sum of all y_tilde_cs: {:.5f}".format(sum(all_ytildecs)))
    # print("suitable alpha value: {:.5f}".format(sum(all_ycs)/sum(all_ytildecs)))
    return alldistance/total, all_kl_distance/total, all_hellinger_distance/total, hits/total, all_bar_yc_distance/total

alldistances = []
all_kl_distances = []
all_hellinger_distances = []
all_accuracies = []
all_barycdistance = []
alphas = np.linspace(0, 1, 101)
for alpha in alphas:
    l1_distance, kl_distance, hellinger_distance, accuracy, barycdistance = run(alpha)
    print(alpha, "{:.5f}".format(l1_distance), "{:.5f}".format(kl_distance), "{:.5f}".format(hellinger_distance), "{:.5f}".format(accuracy), "{:.5f}".format(barycdistance))
    alldistances.append(l1_distance)
    all_kl_distances.append(kl_distance)
    all_hellinger_distances.append(hellinger_distance)
    all_accuracies.append(accuracy)
    all_barycdistance.append(barycdistance)


alldistances = np.array(alldistances)
all_kl_distances = np.array(all_kl_distances)
all_hellinger_distances = np.array(all_hellinger_distances)
all_accuracies = np.array(all_accuracies)
np.save(os.path.join(os.path.dirname(tildefile), "L1_distances_epoch3.npy"), alldistances)
np.save(os.path.join(os.path.dirname(tildefile), "KL_distances_epoch3.npy"), all_kl_distances)
np.save(os.path.join(os.path.dirname(tildefile), "Hellinger_distances_epoch3.npy"), all_hellinger_distances)
np.save(os.path.join(os.path.dirname(tildefile), "all_accuracies_epoch3.npy"), all_accuracies)