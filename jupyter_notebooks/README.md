# Note the notebook refers to local files not in the repository

## Directory Structure

The notebooks refer to directories in `/data` (which should be a virtual directory setup when you run the Docker image).  

In that directory there should be the following directory structure:  
* `/augmentation`
 - `/ngram_dist` - *n-gram distributions of 5 MRQA Training sets.*
 - `/predictions` - *Predictions generated using the augmented test sets*
 - `/test` - *Augmented test sets*
 - `/results` - *Files output from analysis*
* `/ablation`
 - `/models` - *Outputted weights from ablation*
 - `/results` - *Files output from analysis*


### Jupyter notebooks  

#### [Generate SQuAD Test Set Augmentation.ipynb](Generate%20SQuAD%20Test%20Set%20Augmentation.ipynb)

Using the 4 Miller et. al. test sets as source augments the test sets by masking words based on either part of speech or frequency it appears in Wikipedia and predicts a replacement.  

#### [Generate Augmented SQuAD Predictions](Generate%20Augmented%20SQuAD%20Predictions.ipynb)

Predict the answers from the augmented test sets generated in `Generate SQuAD Test Set Augmentation.ipynb`.  
Output: The predictions to the container dir `/data/augmentation/predictions/` (subsequently uploaded to AWS S3).

#### [Generate SQuAD Eval Score](Generate%20SQuAD%20Eval%20Score.ipynb)

Scores all of the predictions against the augmented test sets and generates a matrix of the results at both the model level an question level.  
Output: The score matrices `question_results.parquet.gz` & `model_results.parquet.gz` to the container dir `/data/augmentation/results/`  (subsequently uploaded to AWS S3).

#### [Analyze Augmentation Result](Analyze%20Augmentation%20Result.ipynb)

Analysis of augmented model and question scores.


#### [Generate n-gram Distributions](Generate%20n-gram%20Distributions.ipynb)

Generates n-gram distributions of 6 training sets made available through MRQA.
Output: Pickled NLTK fdist objects to the container's `/data/augmentation/ngram_dist/model_scores.parquet.gz` directory.  


### [Analyze n-gram Distributions](Analyze%20n-gram%20Distributions.ipynb)

Analysis of n-gram frequency distributions and augmented model scores.
No output, but this can use huge amounts of RAM, so might be advantageous to output scoring results.

#### [Analyze Ablation Results](Analyze%20Ablation%20Results.ipynb)

Compiles the scores which have been output in the container's `/data/ablation/models` subdirectories by running `eval_v1_1.py` on the command line.
Output: Matrix of model scores stored in `/data/ablation/results`




## Docker Stuff

```
$ docker run --rm -it --gpus=all -p 8888:8888 -v $(pwd):/workspace -v /data:/data kevinhanna/w210_nlp_jupyter
```

Notes:
- For windows replace `$(pwd)` with `${PWD}` (or to the path to the `jupyter_notebooks` directory)
- Change the first `/data` to where you want to store output.


Once the docker container is running, start Jupyter:  
Within the `/workspace` directory (default) run the alias `$ jnote`.

Aliases in docker image:
- `jnote` - Starts the Jupyter Notebook server and ouputs the login url.
- `jtoken` - Finds and outputs the session token for Jupyter.

Alternatively:
```
$ nohup jupyter notebook --ip=0.0.0.0 --allow-root &  
$ tail nohup.out  
```
There is also an alias `jupnote` for the above command.   
