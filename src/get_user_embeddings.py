import torch
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from sentence_transformers import models, SentenceTransformer
from collections import defaultdict



def main():
    
    data_indices = [0,1,2,3,4]
    
    model_types = ['model_bert_ft', 'model_roberta-base_ft', 'model_bert', 'model_roberta-base']#, 'model_sbert_ft', 'model_sbert'
    model_labels = ['Finetuned BERT', 'Finetuned SBERT', 'BERT', 'SBERT'] #, 'Finetuned SBERT roberta', 'SBERT roberta'
    
    for data_idx in data_indices:
        
        df_train = pd.read_pickle('../dataset/04_K-fold_dataset/df_train_idx%d.p'%(data_idx))
        df_test  = pd.read_pickle('../dataset/04_K-fold_dataset/df_test_idx%d.p'%(data_idx))
    
    
        #user data from train set 
        df_g = df_train.groupby("username")
        
        belief_sequences = []
        user2beliefs_train = {}
        
        for g, data in df_g:
            sequence = list(data['belief_statement'].unique())
            user2beliefs_train[g] = sequence
            belief_sequences.append(sequence)   
        
        seq_sizes = [len(e) for e in belief_sequences]
        print("Dataidx(%d) Train sequence 수:%d"%(data_idx, len(seq_sizes)), "average length: %.2f" % np.mean(seq_sizes))


        with open('../dataset/UserEmbeddings/user2beliefstatements_data(%d).p'%(data_idx), 'wb' ) as f :   
            pickle.dump(user2beliefs_train, f) #all beliefs of each user in the train data

        
        #belief embedding - train for models
        model2user2embeddings_train = {}
        
        for model_type in model_types:
        
            
            belief_vector_path = '../dataset/BeliefEmbeddingResults/BeliefEmbeddings_data(%d)_model(%s).p'%(data_idx, model_type)
            with open(belief_vector_path, 'rb') as f:
                train_belief2embeddings = pickle.load(f) #belief statement : belief vector dict. 


            model2user2embeddings_train[model_type] = defaultdict(list)
            for user in user2beliefs_train:
                for belief in user2beliefs_train[user]:
                    model2user2embeddings_train[model_type][user].append(train_belief2embeddings[belief])
                    #contains list of belief vectors of users

                    
            with open('../dataset/UserEmbeddings/user2beliefvectors_data(%d)_model(%s).p'%(data_idx, model_type), 'wb' ) as f :   
                pickle.dump(model2user2embeddings_train[model_type], f)

        
        #vote_history
        for model_type in model_types:
        
            user2embeddings_train = model2user2embeddings_train[model_type]
        
            #user vote history
            user2length_train = {}
            for e in user2embeddings_train:
                user2length_train[e] = len(user2embeddings_train[e])
            
            
            #User embedding - Average belief embedding     
            user2avg_emb_train = {}
            userlist_train = []
            for e in user2embeddings_train:
                userlist_train.append(e)
                user2avg_emb_train[e] = np.mean(np.array(user2embeddings_train[e]), axis=0) #user position
            
            
            with open('../dataset/UserEmbeddings/user2embeddingvector_data(%d)_model(%s).p'%(data_idx, model_type), 'wb' ) as f:   
                pickle.dump(user2avg_emb_train, f)

            

if __name__ == "__main__":
    main()