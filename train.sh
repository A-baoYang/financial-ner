MODEL_TYPE=chinese_bert_wwm
DATASET=gpt35_filtered
BATCH_SIZE=16

python main.py --task $DATASET --model_type $MODEL_TYPE --model_dir bert_crf_label_$DATASET --do_train --do_eval --use_crf --batch_size=$BATCH_SIZE