#!/usr/bin/env bash

datapath=/home/dev/xiongjun/gif_classify/gif_classify/records_data/train
eval_path=/home/dev/xiongjun/gif_classify/gif_classify/records_data/test

model_name=NeXtVLADModel2
parameters="--groups=8 --nextvlad_cluster_size=128 --nextvlad_hidden_size=1024 \
            --expansion=2 --gating_reduction=8 --drop_rate=0.5"

# train_dir=nextvlad_8g_5l2_5drop_128k_2048_2x80_logistic
train_dir=gif_model_logistic
result_folder=gif_results

echo "model name: " $model_name
echo "model parameters: " $parameters

echo "training directory: " $train_dir
echo "data path: " $datapath
echo "evaluation path: " $eval_path

python gif_train.py ${parameters} --model=${model_name}  --num_readers=8 --learning_rate_decay_examples 2000000 \
                --video_level_classifier_model=LogisticModel --label_loss=CrossEntropyLoss --start_new_model=True \
                --train_data_pattern=${datapath}/*.tfrecord --train_dir=${train_dir} --frame_features=True \
                --feature_names="rgb" --feature_sizes="1024" --batch_size=80 --base_learning_rate=0.0002 \
                --learning_rate_decay=0.8 --l2_penalty=7e-4 --max_step=700000 --num_gpu=1 --num_epochs=30

# python gif_eval.py ${parameters} --batch_size=80 --video_level_classifier_model=LogisticModel --l2_penalty=1e-5\
#                --label_loss=CrossEntropyLoss --eval_data_pattern=${eval_path}/*.tfrecord --train_dir ${train_dir} \
#                --run_once=True

# mkdir -p $result_folder
# python inference.py --output_model_tgz ${result_folder}/${train_dir}.tgz \
#                     --output_file ${result_folder}/${train_dir}.csv \
#                     --input_data_pattern=${test_path}/test*.tfrecord --train_dir ${train_dir} \
#                     --batch_size=80 --num_readers=8
