export CUDA_VISIBLE_DEVICES=1

pip install scikit-learn
pip install numpy==1.26.4

probetype="mlp"
threshold_yc=0.5
# expdir=exp/qwen25_3B_instruct_contaminate_mathmcqa_with_indirect_eval_order1_unlearn_ensemble_${threshold_yc}
expdir=exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_eval_unlearn_ensemble_${threshold_yc}

# modelname=Qwen/Qwen2.5-3B-Instruct
modelname=meta-llama/Llama-3.2-3B-Instruct

mkdir -p $expdir

# traindata=exp/llama32_3B_instruct_contaminate_mmlupro/mmlupro_target_results_with_bar_y_orig.json
# traindata=exp/llama32_3B_instruct_contaminate_mmlupro_indirect/mmlupro_target_results_with_tilde_y_epoch1.json
# validdata=exp/llama32_3B_instruct_contaminate_mmlupro_indirect/mmlupro_target_results_with_bar_y_epoch1_orig.json
# traindata=exp/llama32_3B_instruct_contaminate_mmlupro_true_eval/mmlupro_target_results_with_bar_y_epoch5_devtilde_ytilde.json
# traindata=exp/llama32_3B_instruct_contaminate_mmlupro_with_indirect_eval_redosmall/mmlupro_target_results_with_bar_y_epoch5_truedeveval_origmodel.json
# traindata=exp/llama32_3B_instruct_contaminate_mathmcqa_with_indirect_eval_order1/mathmcqa_target_results_with_bar_y_epoch5_truedeveval_origmodel.json
# traindata=exp/qwen25_3B_instruct_contaminate_mmlupro_with_indirect_eval_order1/mmlupro_target_results_with_bar_y_epoch5_truedeveval_origmodel.json
traindata=results/ensemble_biased_model_truedeveval.json

# validdata=exp/qwen25_3B_instruct_contaminate_mathmcqa_with_indirect_dev_order1/mathmcqa_target_results_with_bar_y_epoch5_trueeval_ensemble.json
validdata=exp/qwen25_3B_instruct_contaminate_mmlupro_with_indirect_dev_order1/mmlupro_target_results_with_bar_y_epoch5_trueeval_ensemble.json
# load_from=exp/llama32_3B_instruct_contaminate_mmlupro_eval/checkpoint.2.final
load_from=exp/qwen25_3B_instruct_contaminate_mathmcqa_with_indirect_eval_order1/checkpoint.4.final

python train.py \
    --model_path $modelname \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10 \
    --num_warmup_steps 0.0 \
    --weight_decay 0.0 \
    --lr_scheduler_type constant \
    --outputdir $expdir \
    --logfile $expdir/log.txt \
    --log_interval 200 \
    --save_interval 20000 \
    --train_data_path $traindata \
    --val_data_path $validdata \
    --lora_config ./config/lora_config.json \
    --load_from $load_from \
    --unlearnmode \
    --losstype fixed_kl_norm \
    --threshold_yc $threshold_yc \
    # --probetype $probetype \
    # --probelayer -1 \
    # --unlearnmode \
    # --losstype fixed_kl_norm \
    # --alpha 0.9 \