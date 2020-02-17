#!/usr/bin/env bash

datapath=/home/dev/xiongjun/gif_classify/gif_classify/records_data/train
eval_path=/home/dev/xiongjun/gif_classify/gif_classify/records_data/test

model_name=NeXtVLADModel2
parameters="--groups=8 --nextvlad_cluster_size=128 --nextvlad_hidden_size=2048 \
            --expansion=2 --gating_reduction=8 --drop_rate=0.5"

# train_dir=nextvlad_8g_5l2_5drop_128k_2048_2x80_logistic
train_dir=gif_model_logistic
result_folder=gif_results

echo "model name: " $model_name
echo "model parameters: " $parameters

echo "training directory: " $train_dir
echo "data path: " $datapath
echo "evaluation path: " $eval_path

python gif_eval.py ${parameters} --batch_size=80 --video_level_classifier_model=LogisticModel --l2_penalty=7e-5\
               --label_loss=CrossEntropyLoss --eval_data_pattern=${eval_path}/*.tfrecord --train_dir ${train_dir} \
               --run_once=True

