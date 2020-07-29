export MRQA_DIR=/home/ubuntu/mids/data/qa/mrqa-latest
python run_mrqa.py \
--model_type mrqa \
--model_name_or_path bert-large-uncased-whole-word-masking \
--do_eval \
--do_lower_case \
--overwrite_cache \
--predict_file $MRQA_DIR/mrqa-dev.json \
--train_file $MRQA_DIR/mrqa-train.json \
--output_dir ../models/mrqa/ \
