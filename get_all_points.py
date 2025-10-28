import json
import numpy as np
from scipy import stats
# from sklearn.linear_model import LinearRegression
from scipy.stats import linregress


unbiased_eval = "exp/llama32_3B_instruct_contaminate_mmlupro_true_dev/mmlupro_target_results_with_bar_y_epoch5_trueeval.json"
biased_eval = "exp/llama32_3B_instruct_contaminate_mmlupro_true_eval/mmlupro_target_results_with_bar_y_epoch5_trueeval.json"
biased_ytilde = "exp/llama32_3B_instruct_contaminate_mmlupro_true_eval/mmlupro_target_results_with_bar_y_epoch5_devtilde_ytilde.json"

# unbiased_eval = "exp/Intern3_8B_instruct_contaminate_mmlupro_true_dev/mmlupro_target_results_with_bar_y_epoch5_trueeval.json"
# biased_eval = "exp/Intern3_8B_instruct_contaminate_mmlupro_true_eval/mmlupro_target_results_with_bar_y_epoch5_trueeval.json"
# biased_ytilde = "exp/Intern3_8B_instruct_contaminate_mmlupro_true_eval/mmlupro_target_results_with_bar_y_epoch5_devtilde.json"


letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
all_D_eval_theta_bias_c = []
all_D_eval_theta_bias_d = []
all_D_eval_theta_unbias_c = []
all_D_eval_theta_unbias_d = []
all_D_dev_tilde_bias_c = []
# D eval theta bias
with open(biased_eval) as fin:
    D_eval_theta_bias = json.load(fin)

all_points = {}

for datapiece in D_eval_theta_bias:
    if datapiece["question_id"] not in all_points:
        all_points[datapiece["question_id"]] = {}
    yc_index = letters.index(datapiece["answer"])
    sum_bar_y = sum(datapiece["bar_y"])
    datapiece["bar_y"] = [yi/sum_bar_y for yi in datapiece["bar_y"]]
    all_points[datapiece["question_id"]]["D_eval_theta_bias_c"] = datapiece["bar_y"][yc_index]
    all_D_eval_theta_bias_c.append(datapiece["bar_y"][yc_index])
    all_points[datapiece["question_id"]]["D_eval_theta_bias_d"] = datapiece["bar_y"][:yc_index] + datapiece["bar_y"][yc_index+1:]
    all_D_eval_theta_bias_d.extend(datapiece["bar_y"][:yc_index] + datapiece["bar_y"][yc_index+1:])

with open(unbiased_eval) as fin:
    D_eval_theta_unbias = json.load(fin)

for datapiece in D_eval_theta_unbias:
    yc_index = letters.index(datapiece["answer"])
    sum_bar_y = sum(datapiece["bar_y"])
    datapiece["bar_y"] = [yi/sum_bar_y for yi in datapiece["bar_y"]]
    all_points[datapiece["question_id"]]["D_eval_theta_unbias_c"] = datapiece["bar_y"][yc_index]
    all_D_eval_theta_unbias_c.append(datapiece["bar_y"][yc_index])
    all_points[datapiece["question_id"]]["D_eval_theta_unbias_d"] = datapiece["bar_y"][:yc_index] + datapiece["bar_y"][yc_index+1:]
    all_D_eval_theta_unbias_d.extend(datapiece["bar_y"][:yc_index] + datapiece["bar_y"][yc_index+1:])

with open(biased_ytilde) as fin:
    D_dev_tilde_bias = json.load(fin)

for datapiece in D_dev_tilde_bias:
    if "tildeyc" in datapiece:
        all_points[datapiece["question_id"]]["D_dev_tilde_bias_c"] = datapiece["tildeyc"]
        all_D_dev_tilde_bias_c.append(datapiece["tildeyc"])
    else:
        yc_index = letters.index(datapiece["answer"])
        all_points[datapiece["question_id"]]["D_dev_tilde_bias_c"] = datapiece["bar_y"][yc_index]
        all_D_dev_tilde_bias_c.append(datapiece["bar_y"][yc_index])

# with open("all_points.json", "w") as fout:
#     json.dump(all_points, fout, indent=4)

all_D_eval_theta_bias_c = np.array(all_D_eval_theta_bias_c)
all_D_eval_theta_bias_d = np.array(all_D_eval_theta_bias_d)
all_D_eval_theta_unbias_c = np.array(all_D_eval_theta_unbias_c)
all_D_eval_theta_unbias_d = np.array(all_D_eval_theta_unbias_d)
all_D_dev_tilde_bias_c = np.array(all_D_dev_tilde_bias_c)

src_D_eval_theta_bias_c_D_eval_theta_unbias_c = stats.spearmanr(all_D_eval_theta_bias_c, all_D_eval_theta_unbias_c)
src_all_D_dev_tilde_bias_c_all_D_eval_theta_unbias_c = stats.spearmanr(all_D_dev_tilde_bias_c, all_D_eval_theta_unbias_c)
src_all_D_eval_theta_bias_d_all_D_eval_theta_unbias_d = stats.spearmanr(all_D_eval_theta_bias_d, all_D_eval_theta_unbias_d)
print("SRC D_eval_theta_bias vs. D_eval_theta_unbias: {:.3f}".format(src_D_eval_theta_bias_c_D_eval_theta_unbias_c.correlation))
print("SRC D_dev_tilde_theta_bias vs. D_eval_theta_unbias: {:.3f}".format(src_all_D_dev_tilde_bias_c_all_D_eval_theta_unbias_c.correlation))
print("SRC D_eval_theta_bias_distractors vs. D_eval_theta_unbias_distractors: {:.3f}".format(src_all_D_eval_theta_bias_d_all_D_eval_theta_unbias_d.correlation))

pcc_D_eval_theta_bias_c_D_eval_theta_unbias_c = stats.pearsonr(all_D_eval_theta_bias_c, all_D_eval_theta_unbias_c)
pcc_all_D_dev_tilde_bias_c_all_D_eval_theta_unbias_c = stats.pearsonr(all_D_dev_tilde_bias_c, all_D_eval_theta_unbias_c)
pcc_all_D_eval_theta_bias_d_all_D_eval_theta_unbias_d = stats.pearsonr(all_D_eval_theta_bias_d, all_D_eval_theta_unbias_d)

a, b = np.polyfit(all_D_eval_theta_bias_c, all_D_eval_theta_unbias_c, 1)
slope = np.linalg.lstsq(all_D_eval_theta_bias_c[:, None], all_D_eval_theta_unbias_c, rcond=None)[0][0]
print("PCC D_eval_theta_bias vs. D_eval_theta_unbias: {:.3f}, a = {:.3f}, b = {:.3f}, slope = {:.3f}".format(pcc_D_eval_theta_bias_c_D_eval_theta_unbias_c[0], a, b, slope))
print("PCC D_dev_tilde_theta_bias vs. D_eval_theta_unbias: {:.3f}".format(pcc_all_D_dev_tilde_bias_c_all_D_eval_theta_unbias_c[0]))
a, b = np.polyfit(all_D_eval_theta_bias_d, all_D_eval_theta_unbias_d, 1)
# a, b = np.polyfit(all_D_eval_theta_unbias_d, all_D_eval_theta_bias_d, 1)
slope = np.linalg.lstsq(all_D_eval_theta_bias_d[:, None], all_D_eval_theta_unbias_d, rcond=None)[0][0]
print("PCC D_eval_theta_bias_distractors vs. D_eval_theta_unbias_distractors: {:.3f}, a = {:.3f}, b = {:.3f}, slope = {:.3f}".format(pcc_all_D_eval_theta_bias_d_all_D_eval_theta_unbias_d[0], a, b, slope))

# np.save("/scratch/DataContamination/to_plot/all_D_eval_theta_bias_c.npy", all_D_eval_theta_bias_c)
# np.save("/scratch/DataContamination/to_plot/all_D_eval_theta_bias_d.npy", all_D_eval_theta_bias_d)
# np.save("/scratch/DataContamination/to_plot/all_D_eval_theta_unbias_c.npy", all_D_eval_theta_unbias_c)
# np.save("/scratch/DataContamination/to_plot/all_D_eval_theta_unbias_d.npy", all_D_eval_theta_unbias_d)
# np.save("/scratch/DataContamination/to_plot/all_D_dev_tilde_bias_c.npy", all_D_dev_tilde_bias_c)