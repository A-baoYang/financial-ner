## Version
    python3
    pytorch>=1.2
## Installation
    pip install transformers,pytorch-crf,seqeval
## How to train the model
    python main.py --task weibo --model_type bert --model_dir weibo_model --do_train --do_eval (--use_crf)
    python main.py --task ltn_news --model_type bert --model_dir ltn_model_crf_alldata --do_train --do_eval --use_crf
    python main.py --task gpt35 --model_type bert --model_dir bert_crf_label_gpt35 --do_train --do_eval --use_crf
## How to predict the new data
    python main.py --task weibo --model_type bert --model_dir weibo_model --do_pred (--use_crf) --pred_dir ./preds/<project_dir> --pred_input_file preds.txt --pred_output_file outputs.txt
## Some question about dataset
the `Weibo` dataset we use has some questions, you can download it in [here](https://github.com/hltcoe/golden-horse/tree/master/data)
## The explanation of do_pred.py
you can use it to reply the "POST" request (through the package flask).
