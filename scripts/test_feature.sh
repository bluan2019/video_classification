#!/usr/bin/env bash

model_dir="/home/dev/xiongjun/yt8m_model_data"
input_csv="test.txt"
output="test_out.tfrecord"
fms=10

python feature_extractor/extract_gif_features.py --output_tfrecords_file=${output} --input_videos_csv=${input_csv} --model_dir=${model_dir} --frames_per_second=${fms} --insert_zero_audio_features=False 

