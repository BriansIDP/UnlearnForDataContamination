. /research/milsrg1/user_workspace/gs534/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate unlearn

expdir=exp/llama32_3B_instruct_contaminate_mmlupro_true_eval
# expdir=exp/llama32_3B_instruct_contaminate_mmlupro_unlearn_devtilde_ytilde0.9_fixed_bary_norm
# expdir=exp/llama32_3B_instruct_contaminate_mmlupro_unlearn_oracle
modelpath=$expdir
# modelpath=meta-llama/Llama-3.2-3B-Instruct
# modelpath=microsoft/Phi-3.5-mini-instruct
ckpt=checkpoint.4.final
testdata=data/train_target_true_eval.json
# testdata=data/mmlupro_alt_dev_6.json
# testdata=data/mmlupro_alt_dev_tilde_1.json

python inference.py \
    --model_path $modelpath \
    --model_ckpt $ckpt \
    --testfile $testdata \
    --outfile $expdir/mmlupro_target_results_with_bar_y_epoch5_trueeval_losscurve1iter.json \
    --nsamples 1 \
    --do_generation \
    --allchoices \
    --get_movements \
    # --origmodel \
    # --outputlogp \
    # --allchoices \
    # --origmodel \


# expdir=exp/Intern3_8B_instruct_contaminate_mmlupro_true_eval
# modelpath=$expdir
# # modelpath=meta-llama/Llama-3.2-3B-Instruct
# ckpt=checkpoint.4.final
# testdata=data/train_target_true_eval.json

# python inference.py \
#     --model_path $modelpath \
#     --model_ckpt $ckpt \
#     --testfile $testdata \
#     --outfile $expdir/mmlupro_target_results_with_bar_y_epoch5_truedev.json \
#     --nsamples 1 \
#     --do_generation \
#     --allchoices \
#     # --origmodel \
#     # --outputlogp \

# expdir=exp/llama32_3B_instruct_contaminate_mmlupro_true_eval
# modelpath=$expdir
# # modelpath=meta-llama/Llama-3.2-3B-Instruct
# ckpt=checkpoint.4.final
# testdata=data/mmlupro_alt_dev_tilde_1.json

# python inference.py \
#     --model_path $modelpath \
#     --model_ckpt $ckpt \
#     --testfile $testdata \
#     --outfile $expdir/mmlupro_target_results_with_bar_y_epoch5_devtilde_ytilde.json \
#     --nsamples 1 \
#     --do_generation \
#     --outputlogp \
#     # --allchoices \
#     # --origmodel \
#     # --outputlogp \

# expdir=exp/llama32_3B_instruct_contaminate_mmlupro_true_eval
# modelpath=$expdir
# # modelpath=meta-llama/Llama-3.2-3B-Instruct
# ckpt=checkpoint.4.final
# testdata=data/alternative_dev_bar_1.json

# python inference.py \
#     --model_path $modelpath \
#     --model_ckpt $ckpt \
#     --testfile $testdata \
#     --outfile $expdir/mmlupro_target_results_with_bar_y_epoch5_devbar.json \
#     --nsamples 1 \
#     --do_generation \
#     --allchoices \
#     # --origmodel \
#     # --outputlogp \
