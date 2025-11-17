export CUDA_VISIBLE_DEVICES=0

expdir=exp/llama32_3B_instruct_contaminate_mmlupro_eval_probe
# expdir=exp/llama32_3B_instruct_contaminate_mmlupro_unlearn_devtilde_ytilde0.1_fixed_bary_norm
# expdir=exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_eval_ytilde0.9_fixed_bary_norm
# modelname=Qwen/Qwen2-7B-Instruct
modelname=meta-llama/Llama-3.2-3B-Instruct
# modelname=meta-llama/Llama-3.1-8B-Instruct
mkdir -p $expdir
traindata=data/train_target_mark.json
# traindata=data/train_target_true_eval.json
# validdata=data/train_target_true_eval.json
validdata=data/train_target_mark.json
# traindata=exp/llama32_3B_instruct_contaminate_mmlupro/mmlupro_target_results_with_bar_y_orig.json
# traindata=exp/llama32_3B_instruct_contaminate_mmlupro_indirect/mmlupro_target_results_with_tilde_y_epoch1.json
# validdata=exp/llama32_3B_instruct_contaminate_mmlupro_indirect/mmlupro_target_results_with_bar_y_epoch1_orig.json
# load_from=exp/llama32_3B_instruct_contaminate_mmlupro_indirect/checkpoint.0.final
# traindata=exp/llama32_3B_instruct_contaminate_mmlupro_true_eval/mmlupro_target_results_with_bar_y_epoch5_devtilde_ytilde.json
# traindata=exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_eval/mmlupro_target_results_with_bar_y_epoch5_trueeval.json
# validdata=exp/llama32_3B_instruct_contaminate_mmlupro/mmlupro_target_results_with_bar_y_orig.json
# validdata=exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_dev/mmlupro_target_results_with_bar_y_epoch5_trueeval.json
# load_from=exp/llama32_3B_instruct_contaminate_mmlupro_true_eval/checkpoint.4.final
load_from=exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_eval/checkpoint.2.final

python train.py \
    --model_path $modelname \
    --batch_size 8 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 20 \
    --num_warmup_steps 0.0 \
    --weight_decay 0.0 \
    --lr_scheduler_type linear \
    --outputdir $expdir \
    --logfile $expdir/log.txt \
    --log_interval 500 \
    --save_interval 20000 \
    --train_data_path $traindata \
    --val_data_path $validdata \
    --lora_config ./config/lora_config.json \
    --load_from $load_from \
    --probe \
    # --unlearnmode \
    # --losstype fixed_kl_norm \
    # --alpha 0.9 \