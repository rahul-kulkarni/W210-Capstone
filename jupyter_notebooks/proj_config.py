from proj_utils import *

parts_of_speech_list = [
    ['JJ', 'VB'],
    ['JJ'],
    ['VB', 'RB'],
    ['VB'],
    ['RB'],
    ['RB', 'RBR', 'RBZ'],
    ['VB', 'VBD', 'VBG', 'VBN', 'VBP'],
    ['RB', 'RBR', 'RBZ', 'VB', 'VBD', 'VBG' 'VBN', 'VBP']
]

augmenting_models = [
    'orig',
    'bert',
    'roberta'
]

qa_models = [
    'bert-large-cased-whole-word-masking-finetuned-squad',
    'bert-large-uncased-whole-word-masking-finetuned-squad',
    'distilbert-base-cased-distilled-squad',
    'distilbert-base-uncased-distilled-squad'
]

frequency_percentiles = [
    0.10,
    0.20,
    0.30,
    0.50
]

qa_urls = {
    "amazon_reviews_v1_0": 'https://ndownloader.figshare.com/files/21500109?private_link=2f119bea3e8d711047ec',
    "reddit_v1_0": 'https://ndownloader.figshare.com/files/21500112?private_link=2f119bea3e8d711047ec',
    "new_wiki_v1.0": 'https://ndownloader.figshare.com/files/21500115?private_link=2f119bea3e8d711047ec',
    "nyt_v1.0": 'https://ndownloader.figshare.com/files/21500118?private_link=2f119bea3e8d711047ec',
}

def get_qa_cache():
    qa_cache = {}
    for name, url in qa_urls.items():
        qa_cache[name] = get_gzip_json_url(url)
    return qa_cache

def show_configs():
    print(f"parts_of_speech_list:\n {parts_of_speech_list}\n")
    print(f"augmenting_models:\n {augmenting_models}\n")
    print(f"qa_models:\n {qa_models}\n")
    print(f"qa_urls:\n {qa_urls}\n")
