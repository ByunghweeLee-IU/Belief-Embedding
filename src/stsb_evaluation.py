import argparse
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

import datasets
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import os




def evaluate_sts_sbert(dataname, n_data = 5, n_epochs = 5):
    """
    dataname = 'usersplit', 'without_politics', 'user_downsample', 'topic_downsample', 'temporal_division', 'full_data', 'full_data_bert'
    """
    data_index_list = np.arange(n_data)
    sts_over_epochs = []
    sts_over_epochs_std = []
    
    #Evaluate sts score

    col_epoch = []
    col_data_idx = []
    col_score = []
    
    #for epoch in range(1,n_epochs+1):
    for epoch in [n_epochs]:
    
        sts_result = []
        
        for data_index in data_index_list:
            
            #Load model 
            if dataname=='full_data_bert':
                modelpath = f'../model/{dataname}/bert-base_idx%d_epoch%d'%(data_index, epoch)
                model = SentenceTransformer(modelpath)        
                
            else:
                modelpath = f'../model/{dataname}/roberta-base_idx%d_epoch%d'%(data_index, epoch)
                model = SentenceTransformer(modelpath)    
                
            score = embedding_evaluator(model)
            sts_result.append(score)

            col_epoch.append(epoch)
            col_data_idx.append(data_index)
            col_score.append(score)
            
            print(f"Data:{dataname}  Epoch:{epoch}  Dataidx:{data_index}  Score:{score}")
            

    df_score = pd.DataFrame({'epoch':col_epoch, 'data_idx':col_data_idx, 'score':col_score})
    return df_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="STS evaluation")
    parser.add_argument('--dataname', type=str, required=True, help="Folder name of the dataset")
    parser.add_argument('--n_data', type=int, default=5, help="Number of data index (k-fold)")
    parser.add_argument('--n_epochs', type=int, default=3, help="Number of epochs of the model")

    args = parser.parse_args()
    
    print(f"dataname: {args.dataname}, n_data: {args.n_data}, n_epochs: {args.n_epochs}")
    
    #os.environ['TRANSFORMERS_CACHE'] = '/data3/bl46/cache'
    
    sts = datasets.load_dataset('glue', 'stsb', split='validation')
    sts = sts.map(lambda x: {'label': x['label'] / 5.0})
    
    samples = []
    for sample in sts:
        samples.append(InputExample(
          texts = [sample['sentence1'], sample['sentence2']],
          label = sample['label']
        ))
        
    embedding_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        samples, write_csv=False
    )
    

    df_score = evaluate_sts_sbert(args.dataname, n_data=args.n_data, n_epochs=args.n_epochs)
    
    os.makedirs(f"../dataset-robust/{args.dataname}/evaluation", exist_ok=True)
    
    print(len(df_score))
    df_score.to_pickle(f"../dataset-robust/{args.dataname}/evaluation/sts_evaluation_scores.p")

    









