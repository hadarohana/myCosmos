"""
Embeddings
"""
import pymongo
from pymongo import MongoClient
import os
import logging
logging.basicConfig(format='%(levelname)s :: %(asctime)s :: %(message)s', level=logging.DEBUG)
import time
import click
from joblib import Parallel, delayed
import pandas as pd
import re
import pickle
from collections import defaultdict
import math
from gensim import simple_preprocess as tokenize
from gensim.models import Word2Vec, FastText # Create FastText model for more accuracy (but slower)
import numpy
import faiss

EMBEDDINGS_WORD2VEC_MODEL_FILE = "w2v_model"
index = None
word2int = None
int2word = None

class SentencesIterator():
    def __init__(self, generator_function):
        self.generator_function = generator_function
        self.generator = self.generator_function()

    def __iter__(self):
        # reset the generator
        self.generator = self.generator_function()
        return self

    def __next__(self):
        result = next(self.generator)
        if result is None:
            raise StopIteration
        else:
            return result

def load_pages(db, buffer_size):
    """
    """
    current_docs = []
    for doc in db.objects.find().batch_size(buffer_size):
        current_docs.append(doc.content)
        if len(current_docs) == buffer_size:
            yield tokenize(current_docs)
            current_docs = []
    yield tokenize(current_docs)


def generate_model(num_processes):
    logging.info('Generating gensim model for all documents')
    start_time = time.time()
    client = MongoClient(os.environ['DBCONNECT'])
    logging.info(f'Connected to client: {client}')
    db = client.pdfs
    sentences = SentencesIterator(load_pages(db, num_processes))
    print('Calculating the embeddings...')
    model = Word2Vec(sentences, size=100, window=10, min_count=3, workers=4)
    print('Saving the model...')
    model.save(EMBEDDINGS_WORD2VEC_MODEL_FILE)
    print('WORD2VEC Model saved.')
    return model

def get_index(num_processes):
    global index
    global word2int
    global int2word
    model = generate_model(num_processes)
    word2int={key:k for k,key in enumerate(model.wv.keys())}  
    int2word ={k:key for k,key in enumerate(model.wv.keys())}
    xb=numpy.array(model.wv.values())
    quantizer = faiss.IndexFlatIP(100) # Inner product cosine similarity
    nlist = 100 # Finetune this number of clusters
    m = 8 # bytes per vector
    index = faiss.IndexIVFPQ(quantizer, 100, nlist, m, 8)
    faiss.normalize_L2(xb)
    index.train(xb)
    index.add(xb) 
    index.nprobe = 5
    # return index, word2int

def query(word, k):
    xq = word2int[word]
    D, I = index.search(xq, k)
    return  [int2word[index] for index in I]

@click.command()
@click.argument('num_processes')
def click_wrapper(num_processes):
    get_index(int(num_processes))
    query(word, k)

if __name__ == '__main__':
    click_wrapper()


