export SQUAD_DIR=~/mids/data/qa/squad-latest/
python -m torch.distributed.launch --nproc_per_node=4 run_squad.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --per_gpu_train_batch_size 12 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 320 \
  --doc_stride 128 \
  --output_dir $SQUAD_DIR/bert-base-uncased-squad_v1
