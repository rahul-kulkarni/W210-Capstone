export MRQA_DIR=~/mids/data/qa/mrqa-latest
python -m torch.distributed.launch --nproc_per_node=4 ./run_mrqa.py \
    --model_type bert \
    --model_name_or_path bert-large-uncased-whole-word-masking  \
    --do_train \
    --do_lower_case \
    --train_file $MRQA_DIR/mrqa-train.json \
    --predict_file $MRQA_DIR/mrqa-dev.json \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ../models/wwm_uncased_finetuned_squad/ \
    --per_gpu_eval_batch_size=3   \
    --per_gpu_train_batch_size=3   \

