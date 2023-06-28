PROJECT_DIR=gpt35

python main.py --task gpt35 --model_type bert --model_dir bert_crf_label_gpt35 --do_pred --use_crf --pred_dir ./preds/$PROJECT_DIR --pred_input_file preds.txt --pred_output_file outputs.txt