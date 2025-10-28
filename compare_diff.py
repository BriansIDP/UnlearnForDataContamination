import json

with open("exp/llama32_3B_instruct_contaminate_mmlupro_true_eval/mmlupro_target_results_with_bar_y_epoch5_truedev.json") as fin:
    biased_dev = json.load(fin)

with open("exp/llama32_3B_instruct_contaminate_mmlupro_true_eval/mmlupro_target_results_with_bar_y_epoch5_truedev_orig.json") as fin:
    unbiased_dev = json.load(fin)

unbiaseddict = {}
for datapiece in unbiased_dev:
    unbiaseddict[datapiece["question_id"]] = datapiece

hit = 0
miss = 0
all_better = []
all_worse = []
for datapiece in biased_dev:
    unbiased_pred = unbiaseddict[datapiece["question_id"]]["pred"]
    answer = datapiece["answer"]
    if answer != unbiased_pred and answer == datapiece["pred"]:
        hit += 1
        all_better.append(
            {
                "question_id": datapiece["question_id"],
                "unbiased_pred_str": unbiaseddict[datapiece["question_id"]]["pred_str"],
                "unbiased_pred": unbiaseddict[datapiece["question_id"]]["pred"],
                "biased_pred_str": datapiece["pred_str"],
                "biased_pred": datapiece["pred"],
                "answer": answer,
            }
        )
    elif answer == unbiased_pred and answer != datapiece["pred"]:
        miss += 1
        all_worse.append(
            {
                "question_id": datapiece["question_id"],
                "unbiased_pred_str": unbiaseddict[datapiece["question_id"]]["pred_str"],
                "unbiased_pred": unbiaseddict[datapiece["question_id"]]["pred"],
                "biased_pred_str": datapiece["pred_str"],
                "biased_pred": datapiece["pred"],
                "answer": answer,
            }
        )

print(hit, miss)
with open("exp/better_after_bias.json", "w") as fout:
    json.dump(all_better, fout, indent=4)

with open("exp/worse_after_bias.json", "w") as fout:
    json.dump(all_worse, fout, indent=4)