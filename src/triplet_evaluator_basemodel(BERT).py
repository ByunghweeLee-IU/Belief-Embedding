import torch
import pickle
from tqdm import tqdm
import numpy as np
import random
import pandas as pd
from sentence_transformers import models, SentenceTransformer
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import TripletEvaluator

#BASE BERT 
for foldidx in range(5):

    print(foldidx)
    
    #Train set
    filepath = '../dataset/04_K-fold_triplets/train_triplet_idx%d.p'%(foldidx)

    with open(filepath,'rb') as f:
        triplet_data = pickle.load(f)

    triplet_data = pd.Series(triplet_data)
    triplet_data = triplet_data.sample(100000, random_state=99).to_list()
    
    train_samples = []
    for e in tqdm(triplet_data):
        train_samples.append(InputExample(texts = e))

    bert = models.Transformer('bert-base-uncased')

    pooler = models.Pooling(
        bert.get_word_embedding_dimension(), #768
        pooling_mode_mean_tokens=True #mean pooling
    )
    model = SentenceTransformer(modules=[bert, pooler])

    #Evaluator, using train samples
    triplet_evaluator = TripletEvaluator.from_input_examples(
        train_samples,
        write_csv=True,
        show_progress_bar=True,
        name = 'bert_untrained_(trainset)_idx%d'%(foldidx)
    )

    output_path = '../eval/'
    triplet_evaluator(model, output_path=output_path)


    #Test set
    print(foldidx)
    filepath = '../dataset/04_K-fold_triplets/test_triplet_idx%d.p'%(foldidx)

    with open(filepath,'rb') as f:
        triplet_data = pickle.load(f)

    triplet_data = pd.Series(triplet_data)
    triplet_data = triplet_data.sample(100000, random_state=99).to_list()
    
    test_samples = []
    for e in tqdm(triplet_data):
        test_samples.append(InputExample(texts = e))

    bert = models.Transformer('bert-base-uncased')

    pooler = models.Pooling(
        bert.get_word_embedding_dimension(), #768
        pooling_mode_mean_tokens=True #mean pooling
    )
    model = SentenceTransformer(modules=[bert, pooler])

    #Evaluator, using train samples
    triplet_evaluator = TripletEvaluator.from_input_examples(
        test_samples,
        write_csv=True,
        show_progress_bar=True,
        name = 'bert_untrained_(testset)_idx%d'%(foldidx)
    )

    output_path = '../eval/'
    triplet_evaluator(model, output_path=output_path)
    
