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

**Note:** The long filenames get truncated, using the -n (name) argument forces the worksheet to be stored with the full name.

From the `/W210-Capstone/scripts/squad_eval_code` dir:
```
$ cl upload src -d "Source scripts for testing predictions."
$ cl upload data/allennlp-bidaf-single-predictions -n "allennlp-bidaf-single-predictions"
$ cl upload data/match-lstm-bi-ans-ptr-boundary-single-predictions -n "match-lstm-bi-ans-ptr-boundary-single-predictions"
$ cl upload data/bidaf-self-attention-single-predictions -n "bidaf-self-attention-single-predictions"
```
More datasets are available in this worksheet under "Evaluate on Amazon". https://worksheets.codalab.org/worksheets/0x412235fe516945fa81d04e6938109f0b

## Execute the tests

Using the UUID for the worksheet created above for the `--host-worksheet-uuid` param:

```
$ python3 src/evaluate_squad_model_predictions.py --experiment-info-path data/squad_models_lite.json --test-set-uuid 0x787eae8f0c7846be852fb010da4fe496 --host-worksheet-uuid 0x9f9f102ba5c049d3b3cce69381dfff9f
```

**Note:** The UUID for the `--test-set-uuid` is the version Kevin created.  It's copied from the paper's website.  The stderr: "Evaluation expects v-1.1, but got dataset with v-1.0" occurred on the worksheet John sent as well.   
