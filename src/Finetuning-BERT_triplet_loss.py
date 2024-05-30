import torch
import pickle
from tqdm import tqdm
import numpy as np

from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers import models, SentenceTransformer
from sentence_transformers import losses


for data_index in range(5): #iteration over K-fold 

    #Load dataset 
    filepath = '../dataset/04_K-fold_triplets/train_triplet_idx%d.p'%(data_index) 

    with open(filepath,'rb') as f:
        triplet_data = pickle.load(f)

    print(f"Total # tripliets : {len(triplet_data)}")

    
    #Make InputExamples to use it as input for Data loader 
    triplets = []
    for e in tqdm(triplet_data):
        triplets.append(InputExample(texts = e))

    #Data Loader 
    batch_size = 32
    loader = DataLoader(triplets, shuffle=True, batch_size=batch_size)    


    #Model preparation - BERT 
    bert = models.Transformer('bert-base-uncased')

    pooler = models.Pooling(
                bert.get_word_embedding_dimension(), #768
                    pooling_mode_mean_tokens=True #mean pooling
                    )
    model = SentenceTransformer(modules=[bert, pooler])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("device: ", model.device)


    #Train model 
    loss = losses.TripletLoss(model)
    epochs = 5
    
    for epoch in range(epochs):
        
        savepath = '../model/finetuned-BERT_idx%d_epoch%d'%(data_index, epoch+1)
        
        model.fit(
            train_objectives=[(loader, loss)],
            epochs=1,
            output_path=savepath,
            show_progress_bar=True
        )
