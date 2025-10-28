import json


bar_file = "exp/llama32_3B_instruct_contaminate_mmlupro_true_eval/mmlupro_target_results_with_bar_y_epoch5_trueeval.json"
tilde_file = "exp/llama32_3B_instruct_contaminate_mmlupro_true_eval/mmlupro_target_results_with_bar_y_epoch5_devtilde_ytilde.json"
outfile = "exp/llama32_3B_instruct_contaminate_mmlupro_true_eval/mmlupro_target_results_with_bar_y_epoch5_devtilde_ytilde_ybar.json"

with open(bar_file) as fin:
    bardata = json.load(fin)

idx_to_data = {}
for datapiece in bardata:
    idx_to_data[datapiece["question_id"]] = datapiece["bar_y"]

with open(tilde_file) as fin:
    tildedata = json.load(fin)
for datapiece in tildedata:
    datapiece["bar_y"] = idx_to_data[datapiece["question_id"]]

with open(outfile, "w") as fout:
    json.dump(tildedata, fout, indent=4)
