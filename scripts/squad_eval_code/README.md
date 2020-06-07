# Instructions for reproducing

## Create new CodaLab worksheet

Create the worksheet for this task, and then switch to the new worksheet.
```
$ cl new repro_test
0x9f9f102ba5c049d3b3cce69381dfff9f

$ cl work repro_test
Switched to worksheet: https://worksheets.codalab.org/worksheets/0x9f9f102ba5c049d3b3cce69381dfff9f (repro_test)
```

## Upload source and data

**Note:** The long filenames get truncated, using the -n (name) argument forces the file to be stored with the full name.

From the `/W210-Capstone/scripts/squad_eval_code` dir:
```
$ cl upload src -d "Source scripts for testing predictions."
$ cl upload data/allennlp-bidaf-single-predictions -n "allennlp-bidaf-single-predictions"
$ cl upload data/match-lstm-bi-ans-ptr-boundary-single-predictions -n "match-lstm-bi-ans-ptr-boundary-single-predictions"
$ cl upload data/bidaf-self-attention-single-predictions -n "bidaf-self-attention-single-predictions"
$ cl upload data/bert-large-cased-whole-word-masking-finetuned-squad-predictions -n "bert-large-cased-whole-word-masking-finetuned-squad-predictions"
$ cl upload data/bert-large-uncased-whole-word-masking-finetuned-squad-predictions -n "bert-large-uncased-whole-word-masking-finetuned-squad-predictions"
$ cl upload data/distilbert-base-cased-distilled-squad-predictions -n "distilbert-base-cased-distilled-squad-predictions"
$ cl upload data/distilbert-base-uncased-distilled-squad-predictions -n "distilbert-base-uncased-distilled-squad-predictions"
```
More datasets are available in this worksheet under "Evaluate on Amazon". https://worksheets.codalab.org/worksheets/0x412235fe516945fa81d04e6938109f0b (expand the packages to find the dependency with '-predictions' at the end of it.)

## Execute the tests

Using the UUID for the worksheet created above for the `--host-worksheet-uuid` param:

```
$ python3 src/evaluate_squad_model_predictions.py --experiment-info-path data/squad_models_lite.json --test-set-uuid 0x787eae8f0c7846be852fb010da4fe496 --host-worksheet-uuid 0x9f9f102ba5c049d3b3cce69381dfff9f
```

**Note:** The UUID for the `--test-set-uuid` is the version Kevin created.  It's the Amazon Dataset copied from the paper's website. I'm not sure why this is needed, the '-predictions' are the output, which is all that's needed to calculate the F1, no predicting is actually happening in this process.

The stderr: "Evaluation expects v-1.1, but got dataset with v-1.0" occurred on the worksheet John sent as well.   


## You're done, and now you're wondering.  WTF just happened there?

`evaluate_squad_model_predictions.py` opens the JSON file passed in the --experiment-info-path argument and iterates through the lines to setup tests.  

It looks to see if there is a file with the mode + "-predictions" in the worksheet (--host-worksheet-uuid) and if it finds one, it executes a public file evaluate-v1.1.py (https://worksheets.codalab.org/bundles/0xbcd57bee090b421c982906709c8c27e1).  This is apparently the "Official evaluation script for v1.1 of the SQuAD dataset."
