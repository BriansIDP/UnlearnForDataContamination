export CUDA_VISIBLE_DEVICES=0

pip install scikit-learn
pip install numpy==1.26.4

probetype="mlp"
threshold_yc=0.5
expdir=exp/qwen25_3B_instruct_contaminate_mmlupro_with_indirect_eval_order1
# modelname=Qwen/Qwen2.5-3B-Instruct
modelname=meta-llama/Llama-3.2-3B-Instruct

mkdir -p $expdir
### Datasets
# Use different order number to change batch ordering, we have 5 different orders
traindata=data/train_target_indirect_biased_order1.json
# Uncomment this and use this data for the uncontaminated model finetuning
# traindata=data/train_target_indirect_unbiased_order1.json
validdata=data/train_target_true_eval.json


python train.py \
    --model_path $modelname \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 5 \
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
    # --load_from $load_from \
    # --unlearnmode \
    # --losstype fixed_kl_norm \
    # --threshold_yc $threshold_yc \
    # --probetype $probetype \
    # --probelayer -1 \
    # --unlearnmode \
    # --losstype fixed_kl_norm \
    # --alpha 0.9 \