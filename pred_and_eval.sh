
pred_script="/workspace/MRQA-Shared-Task-2019/baseline/predict.py"
eval_script="/workspace/MRQA-Shared-Task-2019/mrqa_official_eval.py"

mkdir $1/results/

python $pred_script $1/model.tar.gz https://s3.us-east-2.amazonaws.com/mrqa/data/dev/SQuAD.jsonl.gz $1/results/SQuAD_pred.json --cuda_device $2
python $pred_script $1/model.tar.gz https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NewsQA.jsonl.gz $1/results/NewsQA_pred.json --cuda_device $2
python $pred_script $1/model.tar.gz https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TriviaQA-web.jsonl.gz $1/results/TriviaQA_pred.json --cuda_device $2
python $pred_script $1/model.tar.gz https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SearchQA.jsonl.gz $1/results/SearchQA_pred.json --cuda_device $2
python $pred_script $1/model.tar.gz https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/HotpotQA.jsonl.gz $1/results/HotpotQApred.json --cuda_device $2
python $pred_script $1/model.tar.gz https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz $1/results/NaturalQuestions_pred.json --cuda_device $2
python $pred_script $1/model.tar.gz http://participants-area.bioasq.org/MRQA2019/ $1/results/BioASQ_pred.json --cuda_device $2
python $pred_script $1/model.tar.gz https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/DROP.jsonl.gz $1/results/DROP_pred.json --cuda_device $2
python $pred_script $1/model.tar.gz https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/DuoRC.ParaphraseRC.jsonl.gz $1/results/DuoRC_pred.json --cuda_device $2
python $pred_script $1/model.tar.gz https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/RACE.jsonl.gz $1/results/RACE_pred.json --cuda_device $2
python $pred_script $1/model.tar.gz https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/RelationExtraction.jsonl.gz $1/results/RelationExtraction_pred.json --cuda_device $2
python $pred_script $1/model.tar.gz https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TextbookQA.jsonl.gz $1/results/TextbookQA_pred.json --cuda_device $2

python $eval_script https://s3.us-east-2.amazonaws.com/mrqa/data/dev/SQuAD.jsonl.gz $1/results/SQuAD_pred.json > $1/results/SQuAD_score.json
python $eval_script https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NewsQA.jsonl.gz $1/results/NewsQA_pred.json > $1/results/NewsQA_score.json
python $eval_script https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TriviaQA-web.jsonl.gz $1/results/TriviaQA_pred.json > $1/results/TriviaQA_score.json
python $eval_script https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SearchQA.jsonl.gz $1/results/SearchQA_pred.json > $1/results/SearchQA_score.json
python $eval_script https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/HotpotQA.jsonl.gz $1/results/HotpotQApred.json > $1/results/HotpotQApred_score.json
python $eval_script https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz $1/results/NaturalQuestions_pred.json > $1/results/NaturalQuestions_score.json
python $eval_script http://participants-area.bioasq.org/MRQA2019/ $1/results/BioASQ_pred.json > $1/results/BioASQ_score.json
python $eval_script https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/DROP.jsonl.gz $1/results/DROP_pred.json > $1/results/DROP_score.json
python $eval_script https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/DuoRC.ParaphraseRC.jsonl.gz $1/results/DuoRC_pred.json > $1/results/DuoRC_score.json
python $eval_script https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/RACE.jsonl.gz $1/results/RACE_pred.json > $1/results/RACE_score.json
python $eval_script https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/RelationExtraction.jsonl.gz $1/results/RelationExtraction_pred.json > $1/results/RelationExtraction_score.json
python $eval_script https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TextbookQA.jsonl.gz $1/results/TextbookQA_pred.json > $1/results/TextbookQA_score.json
