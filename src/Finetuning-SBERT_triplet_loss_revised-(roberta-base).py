import torch
import pickle
from tqdm import tqdm
import numpy as np

from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers import models, SentenceTransformer
from sentence_transformers import losses

data_index_list = [0,1,2,3,4]  

for data_index in data_index_list:

    #Load dataset 
    filepath = '../dataset/04_K-fold_triplets/train_triplet_idx%d.p'%(data_index) 

    with open(filepath,'rb') as f:
        triplet_data = pickle.load(f)

    #Make InputExamples to use it as input for Data loader 
    triplets = []
    for e in tqdm(triplet_data):
        triplets.append(InputExample(texts = e))

    #Data Loader 
    batch_size = 32
    loader = DataLoader(triplets, shuffle=True, batch_size=batch_size)    
   
    num_epochs = 5
    for epoch in range(num_epochs):
        
        print("Dataidx: %d, epoch: %d"%(data_index, epoch))
        
        if epoch == 0:
            model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')        
        else:
            model = SentenceTransformer('../model/roberta-base_idx%d_epoch%d'%(data_index, epoch))
            
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)

        loss = losses.TripletLoss(model)
        savepath = '../model/roberta-base_idx%d_epoch%d'%(data_index, epoch+1)
        
        model.fit(
            train_objectives=[(loader, loss)],
            epochs=1,
            output_path=savepath,
            show_progress_bar=True
        )



