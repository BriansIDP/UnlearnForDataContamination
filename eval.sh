export CUDA_VISIBLE_DEVICES=0

expdir=exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_eval_redosmall_unlearn_ensemble_oracle_yc_0.0
# expdir=exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_dev_order1
# expdir=exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_eval_ytilde0.1_fixed_bary_norm
# modelpath=meta-llama/Llama-3.2-3B-Instruct
# modelpath=microsoft/Phi-3.5-mini-instruct
ckpt=checkpoint.4.final
# ckpt=checkpoint.0.20000
testdata=data/train_target_true_eval.json
# testdata=data/train_target_true_dev.json
# testdata=data/mmlupro_alt_dev_tilde_1.json

python inference.py \
    --model_path $expdir \
    --model_ckpt $ckpt \
    --testfile $testdata \
    --outfile $expdir/mmlupro_target_results_with_bar_y_epoch5_trueeval.json \
    --nsamples 1 \
    --do_generation \
    --allchoices \
    # --get_movements \
    # --feature_id 4 \
    # --unfreeze_layers 24,25,26,27 \
    # --origmodel \
    # --outputlogp \
    # --allchoices \
    # --origmodel \

# # ckpt=checkpoint.0.final
# expdir=exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_dev_order2
# testdata=data/train_target_true_dev_alt.json
# python inference.py \
#     --model_path $expdir \
#     --model_ckpt $ckpt \
#     --testfile $testdata \
#     --outfile $expdir/mmlupro_target_results_with_bar_y_epoch5_truedev_alt.json \
#     --nsamples 1 \
#     --do_generation \
#     --allchoices \

# # ckpt=checkpoint.1.20000
# expdir=exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_dev_order3
# testdata=data/train_target_true_dev_alt_shuffle.json
# python inference.py \
#     --model_path $expdir \
#     --model_ckpt $ckpt \
#     --testfile $testdata \
#     --outfile $expdir/mmlupro_target_results_with_bar_y_epoch5_truedev_alt_permute.json \
#     --nsamples 1 \
#     --do_generation \
#     --allchoices \

# # ckpt=checkpoint.1.40000
# expdir=exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_dev_order4
# python inference.py \
#     --model_path $expdir \
#     --model_ckpt $ckpt \
#     --testfile $testdata \
#     --outfile $expdir/mmlupro_target_results_with_bar_y_epoch5_trueeval_permute.json \
#     --nsamples 1 \
#     --do_generation \
#     --allchoices \

# # ckpt=checkpoint.1.final
# expdir=exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_dev_order5
# python inference.py \
#     --model_path $expdir \
#     --model_ckpt $ckpt \
#     --testfile $testdata \
#     --outfile $expdir/mmlupro_target_results_with_bar_y_epoch5_trueeval_permute.json \
#     --nsamples 1 \
#     --do_generation \
#     --allchoices \