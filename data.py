import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

def load_isic_training_data(image_folder, ground_truth_file):
    df_ground_truth = pd.read_csv(ground_truth_file)
    # Category names
    known_category_names = list(df_ground_truth.columns.values[1:9])
    unknown_category_name = df_ground_truth.columns.values[9]
    
    # Add path and category columns
    df_ground_truth['path'] = df_ground_truth.apply(lambda row : os.path.join(image_folder, row['image']+'.jpg'), axis=1)
    df_ground_truth['category'] = np.argmax(np.array(df_ground_truth.iloc[:,1:10]), axis=1)
    return df_ground_truth, known_category_names, unknown_category_name

def load_isic_training_and_out_dist_data(isic_image_folder, ground_truth_file, out_dist_image_folder):
    """ISIC training data and Out-of-distribution data are combined"""
    df_ground_truth = pd.read_csv(ground_truth_file)
    # Category names
    known_category_names = list(df_ground_truth.columns.values[1:9])
    unknown_category_name = df_ground_truth.columns.values[9]
    
    # Add path and category columns
    df_ground_truth['path'] = df_ground_truth.apply(lambda row : os.path.join(isic_image_folder, row['image']+'.jpg'), axis=1)
    
    df_out_dist = get_dataframe_from_img_folder(out_dist_image_folder, has_path_col=True)
    for name in known_category_names:
        df_out_dist[name] = 0.0
    df_out_dist[unknown_category_name] = 1.0
    # Change the order of columns
    df_out_dist = df_out_dist[df_ground_truth.columns.values]

    df_combined = pd.concat([df_ground_truth, df_out_dist])
    df_combined['category'] = np.argmax(np.array(df_combined.iloc[:,1:10]), axis=1)

    category_names = known_category_names + [unknown_category_name]
    return df_combined, category_names

def train_validation_split(df):
    df_train, df_val = train_test_split(df, stratify=df['category'], test_size=0.2, random_state=1)
    return df_train, df_val

def compute_class_weight_dict(df):
    """Compute class weights for weighting the loss function on imbalanced data."""
    class_weights = class_weight.compute_class_weight('balanced', np.unique(df['category']), df['category'])
    class_weight_dict = dict(enumerate(class_weights))
    return class_weight_dict, class_weights

def get_dataframe_from_img_folder(img_folder, has_path_col=True):
    if has_path_col:
        return pd.DataFrame([[Path(x).stem, x] for x in sorted(Path(img_folder).glob('**/*.jpg'))], columns =['image', 'path'])
    else:
        return pd.DataFrame([Path(x).stem for x in sorted(Path(img_folder).glob('**/*.jpg'))], columns =['image'])