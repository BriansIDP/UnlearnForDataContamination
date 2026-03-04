export CUDA_VISIBLE_DEVICES=0

pip install scikit-learn
pip install numpy==1.26.4

probetype="mlp"
threshold_yc=0.0
# expdir=exp/llama32_3B_instruct_contaminate_mmlupro_eval_probe_${probetype}
# expdir=exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_dev_order5
expdir=exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_eval_redosmall_unlearn_ensemble_oracle_${threshold_yc}_longtrain
# expdir=exp/llama31_8B_instruct_contaminate_mmlupro_with_indirect_eval
# expdir=exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_eval_ytilde0.9_fixed_bary_norm
# modelname=Qwen/Qwen2-7B-Instruct
modelname=meta-llama/Llama-3.2-3B-Instruct
# modelname=meta-llama/Llama-3.1-8B-Instruct
mkdir -p $expdir
# traindata=data/train_target_indirect_unbiased_order3.json
# traindata=data/train_target_indirect_biased_open.json
# traindata=data/train_target_indirect_biased_order5.json
validdata=data/train_target_true_eval.json
# validdata=data/train_target_mark.json
# validdata=data/train_target_mark_indirect_alpha.json
# traindata=exp/llama32_3B_instruct_contaminate_mmlupro/mmlupro_target_results_with_bar_y_orig.json
# traindata=exp/llama32_3B_instruct_contaminate_mmlupro_indirect/mmlupro_target_results_with_tilde_y_epoch1.json
# validdata=exp/llama32_3B_instruct_contaminate_mmlupro_indirect/mmlupro_target_results_with_bar_y_epoch1_orig.json
# load_from=exp/llama32_3B_instruct_contaminate_mmlupro_indirect/checkpoint.0.final
# traindata=exp/llama32_3B_instruct_contaminate_mmlupro_true_eval/mmlupro_target_results_with_bar_y_epoch5_devtilde_ytilde.json
# traindata=exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_eval/mmlupro_target_results_with_bar_y_epoch5_trueeval.json
traindata=exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_dev/mmlupro_target_results_with_bar_y_epoch5_trueeval_ensemble.json
# traindata=results/ensemble_biased_model_trueeval.json
# validdata=exp/llama32_3B_instruct_contaminate_mmlupro/mmlupro_target_results_with_bar_y_orig.json
validdata=exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_dev/mmlupro_target_results_with_bar_y_epoch5_trueeval_ensemble.json
# validdata=results/ensemble_biased_model_trueeval.json
# load_from=exp/llama32_3B_instruct_contaminate_mmlupro_eval/checkpoint.2.final
load_from=exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_eval_redosmall/checkpoint.4.final

python train.py \
    --model_path $modelname \
    --batch_size 2 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 20 \
    --num_warmup_steps 0.0 \
    --weight_decay 0.0 \
    --lr_scheduler_type constant \
    --outputdir $expdir \
    --logfile $expdir/log.txt \
    --log_interval 200 \
    --save_interval 20000 \
    --train_data_path $traindata \
    --val_data_path $validdata \
    --lora_config ./config/lora_config_unlearn.json \
    --load_from $load_from \
    --unlearnmode \
    --losstype fixed_kl_norm \
    --threshold_yc $threshold_yc \
    # --probetype $probetype \
    # --probelayer -1 \
    # --unlearnmode \
    # --losstype fixed_kl_norm \
    # --alpha 0.9 \