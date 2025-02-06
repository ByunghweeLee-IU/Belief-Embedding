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
import os

def evaluate_model(dataname, trainset_name, testset_name, n_data=5, num_epochs=3, base_model='SBERT'):
    """
    E.g.
        dataname = 'groupkfold'
        modelpath = '../model/'
        trainset_name = 'train_triplet' (trainset file should be in the format "[trainset_name]_idx%d.p" )
        testset_name = 'test_triplet' (testset file should be in the format "[testset_name]_idx%d.p" )
        outputpath = '../dataset-robustness/usersplit/evaluation'
        - outputpath is the folder where the fine-tuned model will be saved. 
        base_model='SBERT' or 'BERT', base model type
    """
    
    datapath = f'../dataset-robust/{dataname}/'
    model_basepath = f'../model/{dataname}/' 
    
    outputpath = f'../dataset-robust/{dataname}/evaluation/'
    os.makedirs(outputpath, exist_ok=True)
    
    data_index_list = np.arange(n_data)
    #epochs = np.arange(1, num_epochs+1)
    epochs = [num_epochs]

    for data_index in data_index_list:
        for epoch in epochs:
    
            print("data idx:", data_index, 'epoch', epoch)
    
            #TRAIN DATSET
            filepath = os.path.join(datapath, trainset_name + '_idx%d.p'%(data_index))
            
            with open(filepath,'rb') as f:
                triplet_data = pickle.load(f)
    
            triplet_data = pd.Series(triplet_data)
            triplet_data = triplet_data.sample(100000, random_state=42).to_list()
            
            train_samples = []
            for e in tqdm(triplet_data):
                train_samples.append(InputExample(texts = e))

            
            ###Test DATSET
            filepath = os.path.join(datapath, testset_name + '_idx%d.p'%(data_index))
    
            with open(filepath,'rb') as f:
                triplet_data = pickle.load(f)
    
            triplet_data = pd.Series(triplet_data)
            triplet_data = triplet_data.sample(min(100000, len(triplet_data)), random_state=42).to_list()
            
            test_samples = []
            for e in tqdm(triplet_data):
                test_samples.append(InputExample(texts = e))
    
                
            
            modelpath = os.path.join(model_basepath, 'roberta-base_idx%d_epoch%d'%(data_index, epoch))
            train_result_name = 'roberta-base-Train_idx%d_epoch%d'%(data_index, epoch)
            test_result_name  = 'roberta-base-Test_idx%d_epoch%d'%(data_index, epoch)
            
            if base_model=='BERT': 
                modelpath = os.path.join(model_basepath, 'bert-base_idx%d_epoch%d'%(data_index, epoch))
                train_result_name = 'bert-base-Train_idx%d_epoch%d'%(data_index, epoch)
                test_result_name  = 'bert-base-Test_idx%d_epoch%d'%(data_index, epoch)

            print(f"modelpath:{modelpath}")
            model = SentenceTransformer(modelpath) 
            print("model loaded")
            
            #Evaluate using train samples
            triplet_evaluator = TripletEvaluator.from_input_examples(
                train_samples,
                write_csv=True,
                show_progress_bar=True,
                name = train_result_name
            )
            triplet_evaluator(model, output_path=outputpath)

            
            #Evaluate using test samples
            triplet_evaluator = TripletEvaluator.from_input_examples(
                test_samples,
                write_csv=True,
                show_progress_bar=True,
                name = test_result_name
            )
            
            triplet_evaluator(model, output_path=outputpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate triplet loss function")
    parser.add_argument('--dataname', type=str, required=True, help="Folder name of the dataset")
    parser.add_argument('--trainset_name', type=str, required=True, help="Train set file name e.g., [trainset_name]_idx[XX].p")
    parser.add_argument('--testset_name', type=str, help="Test set file name e.g., [trainset_name]_idx[XX].p")
    parser.add_argument('--n_data', type=int, default=5, help="Number of data index (k-fold)")
    parser.add_argument('--num_epochs', type=int, default=3, help="Number of epochs of the model")
    parser.add_argument('--base_model', type=str, default='SBERT', help="base llm model: SBERT (default) or BERT")

    args = parser.parse_args()
    
    evaluate_model(args.dataname, args.trainset_name, args.testset_name, args.n_data, args.num_epochs, args.base_model)   
