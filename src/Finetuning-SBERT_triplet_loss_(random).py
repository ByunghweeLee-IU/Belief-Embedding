import torch
import pickle
from tqdm import tqdm
import numpy as np

from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers import models, SentenceTransformer
from sentence_transformers import losses


#Load dataset 
filepath = '../dataset/Randomized_triplets/train_triplet_rand-shuffle.p'

with open(filepath,'rb') as f:
    triplet_data = pickle.load(f)

#Make InputExamples to use it as input for Data loader 
triplets = []
for e in tqdm(triplet_data):
    triplets.append(InputExample(texts = e))

#Data Loader 
batch_size = 32
loader = DataLoader(triplets, shuffle=True, batch_size=batch_size)    

num_epochs = 3
for epoch in range(num_epochs):
    
    print("epoch: %d"%(epoch))
    
    if epoch == 0:
        model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')        
    else:
        model = SentenceTransformer('../model/roberta-base_random(shuffled)_epoch%d'%(epoch))
        
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    loss = losses.TripletLoss(model)
    savepath = '../model/roberta-base_random(shuffled)_epoch%d'%(epoch+1)
    
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=1,
        output_path=savepath,
        show_progress_bar=True
    )



