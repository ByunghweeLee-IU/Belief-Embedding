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


data_index_list = [0,1,2,3,4]
epochs = [1,2,3,4,5]

for data_index in data_index_list:
    for epoch in epochs:

        print("data idx:", data_index, 'epoch', epoch)

        #TRAIN DATSET
        filepath = '../dataset/04_K-fold_triplets/train_triplet_idx%d.p'%(data_index) 

        with open(filepath,'rb') as f:
            triplet_data = pickle.load(f)

        triplet_data = pd.Series(triplet_data)
        triplet_data = triplet_data.sample(100000, random_state=99).to_list()
        
        train_samples = []
        for e in tqdm(triplet_data):
            train_samples.append(InputExample(texts = e))

        #Test DATSET
        filepath = '../dataset/04_K-fold_triplets/test_triplet_idx%d.p'%(data_index) 

        with open(filepath,'rb') as f:
            triplet_data = pickle.load(f)

        triplet_data = pd.Series(triplet_data)
        triplet_data = triplet_data.sample(100000, random_state=99).to_list()
        
        test_samples = []
        for e in tqdm(triplet_data):
            test_samples.append(InputExample(texts = e))

            
        #Load finetuned-BERT model 
        modelpath = '../model/finetuned-BERT_idx%d_epoch%d'%(data_index, epoch)
        model = SentenceTransformer(modelpath)    

        #Evaluate using train samples
        triplet_evaluator = TripletEvaluator.from_input_examples(
            train_samples,
            write_csv=True,
            show_progress_bar=True,
            name = 'finetuned-BERT-Train_idx%d_epoch%d'%(data_index, epoch)
        )
        output_path = '../eval/'
        triplet_evaluator(model, output_path=output_path)

        #Evaluate using test samples
        triplet_evaluator = TripletEvaluator.from_input_examples(
            test_samples,
            write_csv=True,
            show_progress_bar=True,
            name = 'finetuned-BERT-Test_idx%d_epoch%d'%(data_index, epoch)
        )
        output_path = '../eval/'
        triplet_evaluator(model, output_path=output_path)


