import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

def load_isic_data(derm_image_folder, ground_truth_file):
    df_ground_truth = pd.read_csv(ground_truth_file)

    # Category names not include UNK
    category_names = list(df_ground_truth.columns.values[1:9])
    
    # Add path and category columns
    df_ground_truth['path'] = df_ground_truth.apply(lambda row : os.path.join(derm_image_folder, row['image']+'.jpg'), axis=1)
    df_ground_truth['category'] = pd.Series([np.argmax(x) for x in np.array(df_ground_truth.iloc[:,1:9])], name='category')
    return df_ground_truth, category_names

def train_validation_split(df_ground_truth):
    df_train, df_val = train_test_split(df_ground_truth, stratify=df_ground_truth['category'], test_size=0.2, random_state=1)
    return df_train, df_val

def compute_class_weight_dict(df_train):
    """Compute class weights for weighting the loss function on imbalanced data."""
    class_weights = class_weight.compute_class_weight('balanced', np.unique(df_train['category']), df_train['category'])
    class_weight_dict = dict(enumerate(class_weights))
    return class_weight_dict