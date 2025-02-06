import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from collections import defaultdict 
from itertools import combinations

def get_kfold_dataset_from_df(df, savepath, n_splits=5):
    """
        get k fold dataset based on debate titles and save it under savepath folder.              
        input: dataframe (debate, user records)
    """
    
    #shuffle debate titles 
    np.random.seed(42)
    debate_titles = df.debate_title.unique()
    np.random.shuffle(debate_titles)
    print(f"There are {len(debate_titles)} unique debates in debate.org dataset")

    
    #split into K-fold dataset
    kf = KFold(n_splits=5)
    folds = list(kf.split(debate_titles)) #list of indicies of [[[train indices 1], [test indices 1]], ... ] 

    #genenrate train & test dataset 
    fold_data = []

    for i, (train_idx, test_idx) in enumerate(folds):
        
        train_titles = debate_titles[train_idx]
        test_titles = debate_titles[test_idx]
        
        train_df = df[df['debate_title'].isin(train_titles)]
        test_df = df[df['debate_title'].isin(test_titles)]
            
        train_df.to_pickle(savepath + 'df_train_idx%d.p'%(i))
        test_df.to_pickle(savepath + 'df_test_idx%d.p'%(i))
    
        fold_data.append([train_df, test_df])
            
        print(f"Fold {i+1}:")
        print("Train set:")
        print("Debates:", len(train_df), "User:",len(train_df['username'].unique()))
        print("Test set:")
        print("Debates:", len(test_df), "User:", len(test_df['username'].unique()))
        print("="*50)

    return fold_data 


def get_reverse_phrase(phrase):
    pro_phrase = 'I agree with the following: '
    con_phrase = 'I disagree with the following: '
    
    if phrase == pro_phrase:
        return con_phrase
    elif phrase == con_phrase:
        return pro_phrase
    else: 
        print('error')
        
def get_opposite_belief(belief_statement):
    position = " ".join(belief_statement.split()[:5]) + ' '
    title = " ".join(belief_statement.split()[5:]) 
    
    position_r = get_reverse_phrase(position)
    opposite_belief = position_r + title
    return opposite_belief        


#get a belief co-occurrence dictionary
def get_belief_cooccurrence_dic(df):
    
    df_g = df.groupby('username')
    corpus = []

    for g, data in df_g:

        data = data.sort_values(by='debate_date')
        user_beliefs = list(data['belief_statement'].unique())
        corpus.append(user_beliefs)
        
    
    belief2list = defaultdict(list)
    
    for b_list in corpus:
        if len(b_list) == 1: 
            continue

        for e1 in b_list:
            for e2 in b_list:
                if e1 != e2:
                    belief2list[e1].append(e2)
                    
    return belief2list

#Get triplets using the belief co-occurrence dictionary
def get_stance_triplet(belief2list):
    
    belief_triplet = []

    for s in tqdm(belief2list):

        anchor = s
        positive_samples = belief2list[s] 
        opposite_belief = get_opposite_belief(s)

        if not opposite_belief in belief2list: #use only direct opposite stance as a negative sample
            negative_samples = [opposite_belief]
        else:
            negative_samples = [opposite_belief] + belief2list[opposite_belief]

        #if vote history is too long: Sample 5 stances from history 
        thres = 5
        if len(positive_samples) > thres-1:
            positive_samples = np.random.choice(positive_samples, size=thres, replace=False)

        if len(negative_samples) > thres-1:
            #to ensure including directly opposite stance
            other_samples = np.random.choice(negative_samples[1:], size=thres-1, replace=False)        
            negative_samples = np.concatenate((negative_samples[:1], other_samples)) 

        #make triplet examples 
        for pos in positive_samples:
            for neg in negative_samples:
                example = [anchor, pos, neg]
                belief_triplet.append(example)
    
    return belief_triplet

def generate_triplets_from_kfold_data(fold_data, dataset_path, n_splits=5):
    """
    input: kfold dataframe (by default, n_splits=5)
    generate triplets and save for each of k-fold dataset
    """

    for i in range(n_splits):
    
        df_train = fold_data[i][0]  #train set 
        df_test = fold_data[i][1]   #test set   
        
        belief2list_train = get_belief_cooccurrence_dic(df_train)
        belief2list_test  = get_belief_cooccurrence_dic(df_test)
        
        train_triplets = get_stance_triplet(belief2list_train)
        test_triplets  = get_stance_triplet(belief2list_test)
        
        with open(dataset_path + 'train_triplet_idx%d.p'%i,'wb') as f:
            pickle.dump(train_triplets, f)
            
        with open(dataset_path + 'test_triplet_idx%d.p'%i,'wb') as f:
            pickle.dump(test_triplets, f)


def get_common_user_dataset(dataset_path, n_splits=5):
    """
    dataset_path folder contains dataframes: df_train_idx%d.p, df_test_idx%d.p 
    output: generate df_train_common, df_test_common and save them to the dataset_path
    """
    for i in range(n_splits):    
        
        df_train = pd.read_pickle(dataset_path + 'df_train_idx%d.p'%(i))
        df_test  = pd.read_pickle(dataset_path + 'df_test_idx%d.p'%(i))
            
        train_users = df_train.username.unique()
        test_users  = df_test.username.unique()
        
        common_users = []
        for u in test_users:
            if u in train_users:
                common_users.append(u)
        
        df_train_common = df_train[df_train['username'].isin(common_users)]
        df_test_common  = df_test[df_test['username'].isin(common_users)]
                
        df_train_common.to_pickle(dataset_path + 'df_train_common_idx%d.p'%(i))
        df_test_common.to_pickle(dataset_path + 'df_test_common_idx%d.p'%(i))
    
        print("# votes: Train, Train_common, Test, Test_common")
        print(len(df_train), len(df_train_common), len(df_test), len(df_test_common))
        print("# voters: Train, Train_common, Test, Test_common")
        print(len(df_train.username.unique()), len(df_train_common.username.unique()), len(df_test.username.unique()), len(df_test_common.username.unique()))
        print()


def log_binning(data, num_bins=30):
    """
    Parameters:
    - data (list or array): datalist
    - num_bins (int): number of bins on the log-scale axis
    
    Returns:
    - bin_centers (array): midian value of each bin
    - hist (array): number of data in each bin
    """

    min_data, max_data = np.min(data), np.max(data)
    bins = np.logspace(np.log10(min_data), np.log10(max_data), num=num_bins)

    hist, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
    return bin_centers, hist


# Downsample based on user acitivities 
def get_downsampled_data(df, target_col='username', threshold=100):

    target_counts = df[target_col].value_counts()
    to_downsample = target_counts[target_counts>threshold].index #list of users above 100 activities
    
    df_light = df[~df[target_col].isin(to_downsample)]
    df_heavy = df[df[target_col].isin(to_downsample)]
    
    sample_datalist = []
    for e in tqdm(to_downsample):
        sample_datalist.append(df_heavy[df_heavy[target_col]==e].sample(n=threshold, random_state=42))
        
    df_downsampled = pd.concat(sample_datalist)
    df_downsampled = pd.concat([df_light, df_downsampled])
    df_downsampled = df_downsampled.sample(frac=1).reset_index(drop=True)

    return df_downsampled