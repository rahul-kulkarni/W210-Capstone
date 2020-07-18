import gzip
import _pickle as cPickle
from urllib.request import urlopen, Request
import pandas as pd
import json

def get_gzip_pickle(file_url):
    print(f"Fetching: {file_url}")
    with urlopen(Request(file_url,
                     headers={"Accept-Encoding": "gzip"})) as response, \
     gzip.GzipFile(fileobj=response) as myzip:
         unzip_pickle= cPickle.load(myzip)
    
    return unzip_pickle

def get_gzip_parquet(file_url):
    print(f"Fetching: {file_url}")
    unzip_parquet = pd.read_parquet(file_url)
   
    return unzip_parquet

def write_gzip_parquet(dataframe, filepath):
    print("Writting: {}".format(filepath))
    dataframe.to_parquet(filepath,
                  compression='gzip') 
    
def get_json_url(file_url):
    print(f"Fetching: {file_url}")
    
    jsonurl = urlopen(file_url)
    json_results = json.loads(jsonurl.read())
    
    return json_results

def write_gzip_json(file_path, json_data):
    print(f"Writting: {file_path}")
    with gzip.open(file_path, 'wt', encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False)

    return file_path

def get_gzip_json_url(file_url):
    print(f"Fetching: {file_url}")
    with urlopen(Request(file_url,
                     headers={"Accept-Encoding": "gzip"})) as response, \
     gzip.GzipFile(fileobj=response) as myzip:
         unzip_json= json.load(myzip)
    
    return unzip_json

def old_get_gzip_parquet(file_url):
    print(f"Fetching: {file_url}")
    with urlopen(Request(file_url,
                     headers={"Accept-Encoding": "gzip"})) as response, \
     gzip.GzipFile(fileobj=response) as myzip:
         unzip_parquet = pd.read_parquet(myzip)
    
    return unzip_parquet