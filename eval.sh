export CUDA_VISIBLE_DEVICES=1

# expdir=exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_eval_order1
expdir=exp/qwen25_3B_instruct_contaminate_mathmcqa_with_indirect_eval_order1_unlearn_ensemble_0.5
# expdir=exp/llama32_3B_instruct_contaminate_mathmcqa_with_indirect_eval_unlearn_ensemble_0.5
# modelpath=meta-llama/Llama-3.2-3B-Instruct
# modelpath=microsoft/Phi-3.5-mini-instruct
ckpt=checkpoint.9.final
# ckpt=checkpoint.0.20000
testdata=data/MATH_MCQA/math_train_true_eval.json
# testdata=data/train_target_true_dev_shuffle3.json
# testdata=data/train_target_true_dev.json

python inference.py \
    --model_path $expdir \
    --model_ckpt $ckpt \
    --testfile $testdata \
    --outfile $expdir/mathmcqa_target_results_with_bar_y_epoch5_trueeval.json \
    --nsamples 1 \
    --do_generation \
    --allchoices \
    # --origmodel \
    # --get_movements \
    # --feature_id 4 \
    # --unfreeze_layers 24,25,26,27 \
    # --origmodel \
    # --outputlogp \
    # --allchoices \
    # --origmodel \

testdata=data/MATH_MCQA/math_train_true_dev.json
expdir=exp/qwen25_3B_instruct_contaminate_mathmcqa_with_indirect_eval_order1_unlearn_ensemble_0.5
python inference.py \
    --model_path $expdir \
    --model_ckpt $ckpt \
    --testfile $testdata \
    --outfile $expdir/mathmcqa_target_results_with_bar_y_epoch5_truedev.json \
    --nsamples 1 \
    --do_generation \
    --allchoices \

# expdir=exp/qwen25_3B_instruct_contaminate_mathmcqa_with_indirect_dev_order5
# python inference.py \
#     --model_path $expdir \
#     --model_ckpt $ckpt \
#     --testfile $testdata \
#     --outfile $expdir/mathmcqa_target_results_with_bar_y_epoch5_trueeval.json \
#     --nsamples 1 \
#     --do_generation \
#     --allchoices \