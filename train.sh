. /scratch/anaconda/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate ranker

# expdir=exp/llama32_3B_instruct_contaminate_mmlupro_with_dev_2
# expdir=exp/llama32_3B_instruct_contaminate_mmlupro_unlearn_devtilde_ytilde0.1_fixed_bary_norm
expdir=exp/llama32_3B_instruct_contaminate_mmlupro_unlearn_oracle
# modelname=Qwen/Qwen2-7B-Instruct
modelname=meta-llama/Llama-3.2-3B-Instruct
# modelname=meta-llama/Llama-3.1-8B-Instruct
mkdir -p $expdir
# traindata=data/train_target_indirect_exclude.json
# traindata=data/alternative_dev_2.json
# validdata=data/train_target.json
# traindata=exp/llama32_3B_instruct_contaminate_mmlupro/mmlupro_target_results_with_bar_y_orig.json
# traindata=exp/llama32_3B_instruct_contaminate_mmlupro_indirect/mmlupro_target_results_with_tilde_y_epoch1.json
# validdata=exp/llama32_3B_instruct_contaminate_mmlupro_indirect/mmlupro_target_results_with_bar_y_epoch1_orig.json
# load_from=exp/llama32_3B_instruct_contaminate_mmlupro_indirect/checkpoint.0.final
# traindata=exp/llama32_3B_instruct_contaminate_mmlupro_true_eval/mmlupro_target_results_with_bar_y_epoch5_devtilde_ytilde.json
traindata=exp/llama32_3B_instruct_contaminate_mmlupro_true_dev/mmlupro_target_results_with_bar_y_epoch5_trueeval.json
# validdata=exp/llama32_3B_instruct_contaminate_mmlupro/mmlupro_target_results_with_bar_y_orig.json
validdata=exp/llama32_3B_instruct_contaminate_mmlupro_true_dev/mmlupro_target_results_with_bar_y_epoch5_trueeval.json
load_from=exp/llama32_3B_instruct_contaminate_mmlupro_true_eval/checkpoint.4.final

python train.py \
    --model_path $modelname \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 5 \
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
    --unlearnmode \
    --load_from $load_from \
    --losstype fixed_kl_norm \
    --alpha 1.0 \


# expdir=exp/Intern3_8B_instruct_contaminate_mmlupro_true_dev
# # expdir=exp/llama32_3B_instruct_contaminate_mmlupro_indirect_unlearn_ytilde
# # modelname=meta-llama/Llama-3.2-3B-Instruct
# # modelname="microsoft/Phi-4-mini-instruct"
# modelname="internlm/internlm3-8b-instruct"
# mkdir -p $expdir
# # traindata=data/train_target_indirect_exclude.json
# # traindata=data/alternative_dev_6.json
# traindata=data/train_target_true_dev.json
# validdata=data/train_target_true_eval.json


# python train.py \
#     --model_path $modelname \
#     --batch_size 1 \
#     --learning_rate 1e-5 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 5 \
#     --num_warmup_steps 0.0 \
#     --weight_decay 0.0 \
#     --lr_scheduler_type linear \
#     --outputdir $expdir \
#     --logfile $expdir/log.txt \
#     --log_interval 500 \
#     --save_interval 20000 \
#     --train_data_path $traindata \
#     --val_data_path $validdata \
#     --lora_config ./config/lora_config.json \
#     # --unlearnmode \
#     # --load_from $load_from \

# expdir=exp/Phi4_mini_instruct_contaminate_mmlupro_tilde_dev
# # expdir=exp/llama32_3B_instruct_contaminate_mmlupro_indirect_unlearn_ytilde
# # modelname=meta-llama/Llama-3.2-3B-Instruct
# modelname="microsoft/Phi-4-mini-instruct"
# # modelname="internlm/internlm3-8b-instruct"
# mkdir -p $expdir
# # traindata=data/train_target_indirect_exclude.json
# # traindata=data/alternative_dev_6.json
# traindata=data/alternative_dev_tilde_1.json
# validdata=data/train_target_true_eval.json


# python train.py \
#     --model_path $modelname \
#     --batch_size 1 \
#     --learning_rate 1e-5 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 5 \
#     --num_warmup_steps 0.0 \
#     --weight_decay 0.0 \
#     --lr_scheduler_type linear \
#     --outputdir $expdir \
#     --logfile $expdir/log.txt \
#     --log_interval 500 \
#     --save_interval 20000 \
#     --train_data_path $traindata \
#     --val_data_path $validdata \
#     --lora_config ./config/lora_config.json \
#     # --unlearnmode \
#     # --load_from $load_from \


# expdir=exp/Intern_8B_instruct_contaminate_mmlupro_tilde_dev
# # expdir=exp/llama32_3B_instruct_contaminate_mmlupro_indirect_unlearn_ytilde
# # modelname=meta-llama/Llama-3.2-3B-Instruct
# # modelname="microsoft/Phi-4-mini-instruct"
# modelname="internlm/internlm3-8b-instruct"
# mkdir -p $expdir
# # traindata=data/train_target_indirect_exclude.json
# # traindata=data/alternative_dev_6.json
# traindata=data/alternative_dev_tilde_1.json
# validdata=data/train_target_true_eval.json

# python train.py \
#     --model_path $modelname \
#     --batch_size 1 \
#     --learning_rate 1e-5 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 5 \
#     --num_warmup_steps 0.0 \
#     --weight_decay 0.0 \
#     --lr_scheduler_type linear \
#     --outputdir $expdir \
#     --logfile $expdir/log.txt \
#     --log_interval 500 \
#     --save_interval 20000 \
#     --train_data_path $traindata \
#     --val_data_path $validdata \
#     --lora_config ./config/lora_config.json \
#     # --unlearnmode \
#     # --load_from $load_from \

# expdir=exp/Gemma2_9B_instruct_contaminate_mmlupro_tilde_dev
# # expdir=exp/llama32_3B_instruct_contaminate_mmlupro_indirect_unlearn_ytilde
# modelname="google/gemma-2-9b-it"
# # modelname=meta-llama/Llama-3.2-3B-Instruct
# mkdir -p $expdir
# # traindata=data/train_target_indirect_exclude.json
# # traindata=data/alternative_dev_6.json
# traindata=data/alternative_dev_tilde_1.json
# validdata=data/train_target_true_eval.json


# python train.py \
#     --model_path $modelname \
#     --batch_size 1 \
#     --learning_rate 1e-5 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 5 \
#     --num_warmup_steps 0.0 \
#     --weight_decay 0.0 \
#     --lr_scheduler_type linear \
#     --outputdir $expdir \
#     --logfile $expdir/log.txt \
#     --log_interval 500 \
#     --save_interval 20000 \
#     --train_data_path $traindata \
#     --val_data_path $validdata \
#     --lora_config ./config/lora_config.json \
#     # --unlearnmode \
#     # --load_from $load_from \
