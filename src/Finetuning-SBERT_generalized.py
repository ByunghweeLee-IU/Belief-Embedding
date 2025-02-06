import argparse
import torch
import pickle
from tqdm import tqdm
import numpy as np

from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers import models, SentenceTransformer
from sentence_transformers import losses
import os

os.environ['TRANSFORMERS_CACHE'] = '/data3/bl46/cache'

def train_sbert_model(datapath, output_path, num_data=5, num_epochs=3):
    """
        datapath: is the folder where the triplet dataset is.
        e.g. '../dataset-robust/usersplit'
        outputpath is the folder where the fine-tuned model will be saved. 
        e.g. outputpath = '../model/usersplit'
    """
    print(f"Loading data from {datapath}")
    print(f"model output path: {output_path}")
    print(f"Total number of epochs: {num_epochs}")
    
    data_index_list = np.arange(num_data)
    
    for data_index in data_index_list:
        # Load train dataset (triplet data)
        filepath = os.path.join(datapath, f'train_triplet_idx{data_index}.p')
        
        with open(filepath, 'rb') as f:
            triplet_data = pickle.load(f)

        # Make InputExamples and build a dataloader
        triplets = []
        for e in tqdm(triplet_data):
            triplets.append(InputExample(texts=e))

        os.makedirs(output_path, exist_ok=True)
        
        batch_size = 128
        loader = DataLoader(triplets, shuffle=True, batch_size=batch_size)    

        # Train the model
        for epoch in range(num_epochs):            
            print(f"Current data index: {data_index}, epoch: {epoch}")
            
            if epoch == 0:
                model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')        
            else:
                model = SentenceTransformer(os.path.join(output_path, f'roberta-base_idx{data_index}_epoch{epoch}'))

            # Environment setup
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model.to(device)
            
            loss = losses.TripletLoss(model)
            savepath = os.path.join(output_path, f'roberta-base_idx{data_index}_epoch{epoch+1}')
            
            model.fit(
                train_objectives=[(loader, loss)],
                epochs=1,
                output_path=savepath,
                show_progress_bar=True
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Sentence-BERT model on triplet datasets.")
    parser.add_argument('--datapath', type=str, required=True, help="Folder path to the triplet dataset.")
    parser.add_argument('--output_path', type=str, required=True, help="Folder path to save the fine-tuned model.")
    parser.add_argument('--num_data', type=int, default=5, required=True, help="Number of dataset for train (K-fold)")
    parser.add_argument('--num_epochs', type=int, default=3, help="Number of epochs for training.")

    args = parser.parse_args()

    print('start training')
    train_sbert_model(args.datapath, args.output_path, args.num_data, args.num_epochs)
