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

#BASE S-BERT (roberta-base)
for foldidx in range(5):

    #Train set
    filepath = '../dataset/04_K-fold_triplets/train_triplet_idx%d.p'%(foldidx)

    with open(filepath,'rb') as f:
        triplet_data = pickle.load(f)

    triplet_data = pd.Series(triplet_data)
    triplet_data = triplet_data.sample(100000, random_state=99).to_list()
    
    train_samples = []
    for e in tqdm(triplet_data):
        train_samples.append(InputExample(texts = e))

    model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')    

    #Evaluator, using train samples
    triplet_evaluator = TripletEvaluator.from_input_examples(
        train_samples,
        write_csv=True,
        show_progress_bar=True,
        name = 'roberta-base_untrained_(trainset)idx%d'%(foldidx)
    )

    output_path = '../eval/'
    triplet_evaluator(model, output_path=output_path)

    #Test set
    filepath = '../dataset/04_K-fold_triplets/test_triplet_idx%d.p'%(foldidx)

    with open(filepath,'rb') as f:
        triplet_data = pickle.load(f)

    triplet_data = pd.Series(triplet_data)
    triplet_data = triplet_data.sample(100000, random_state=99).to_list()
    
    test_samples = []
    for e in tqdm(triplet_data):
        test_samples.append(InputExample(texts = e))

    model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')    

    #Evaluator, using train samples
    triplet_evaluator = TripletEvaluator.from_input_examples(
        test_samples,
        write_csv=True,
        show_progress_bar=True,
        name = 'roberta-base_untrained_(testset)idx%d'%(foldidx)
    )

    output_path = '../eval/'
    triplet_evaluator(model, output_path=output_path)


