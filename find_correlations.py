import numpy as np
import json
from scipy import stats


with open("exp/llama32_3B_instruct_contaminate_mmlupro_true_eval/mmlupro_target_results_with_bar_y_epoch5_trueeval_losscurve.json") as fin:
    lossdata = json.load(fin)

qid_to_loss = {}
for question in lossdata:
    qid_to_loss[question["question_id"]] = question["loss_curve"]

origfile = "exp/llama32_3B_instruct_contaminate_mmlupro_true_dev/mmlupro_target_results_with_bar_y_epoch5_trueeval.json"
with open(origfile) as fin:
    origdata = json.load(fin)
qid_to_y_orcale = {}
letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
for question in origdata:
    qid_to_y_orcale[question["question_id"]] = question["bar_y"][letters.index(question["answer"])]

evalfile = "exp/llama32_3B_instruct_contaminate_mmlupro_true_eval/mmlupro_target_results_with_bar_y_epoch5_trueeval.json"
with open(evalfile) as fin:
    contam_data = json.load(fin)
qid_to_y_bar = {}
qid_to_alpha = {}
for question in origdata:
    qid_to_y_bar[question["question_id"]] = question["bar_y"][letters.index(question["answer"])]
    qid_to_alpha[question["question_id"]] = qid_to_y_orcale[question["question_id"]] / qid_to_y_bar[question["question_id"]]

all_alphas = []
all_loss_changes = []
for qid, value in qid_to_loss.items():
    all_alphas.append(qid_to_alpha[qid])
    all_loss_changes.append(qid_to_loss[qid])
pcc_alpha_vs_losschange = stats.pearsonr(all_alphas, all_loss_changes)
print("PCC alpha oracle vs. loss change: {:.3f}".format(pcc_alpha_vs_losschange))