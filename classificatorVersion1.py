import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray, rgb2hed
from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.morphology import dilation, erosion, disk, opening, closing, area_closing, area_opening
from skimage.exposure import equalize_hist
from skimage.measure import regionprops
from skimage.measure import label
from scipy import ndimage as ndi
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("train_metadata.csv")
test_df = pd.read_csv("test_metadata.csv")

train_files = train_df["ID"].to_list()
labels_train = train_df["label"].to_list()
test_files = test_df["ID"].to_list()
label_dict = dict(zip(train_df["ID"], train_df["label"]))
def multi_otsu_method(X):
    """Segmentation Part"""
    X_hed = rgb2hed(ndi.gaussian_filter(X, 1.0))
    nuclei_channel = X_hed[:, :, 0] # Canal Hématoxyline

    thresholds = threshold_multiotsu(nuclei_channel, classes=3)
    regions = np.digitize(nuclei_channel, bins=thresholds)
    regions = erosion(regions, disk(5))

    white_cell_mask = (regions > 0) 
    nucleus_mask = (regions > 1)

    labels_cell = label(white_cell_mask)

    touching = np.unique(labels_cell[nucleus_mask])
    touching = touching[touching!=0]
    white_blood_cell = np.isin(labels_cell, touching)
    return white_blood_cell, nucleus_mask

def extract_features(white_blood_cell, nucleus_mask):
    cell_props = regionprops(white_blood_cell.astype(int))
    if len(cell_props) == 0:
        return {
            'area_ratio': 0,
            'cell_eccentricity': 0,
            'num_lobes': 0,
            'nucleus_circularity': 0
        }
    cell_props = max(cell_props, key=lambda x: x.area)
    conv_comp_nucleus = label(nucleus_mask)
    nuc_props = regionprops(conv_comp_nucleus.astype(int))
    nuc_total_area = sum([prop.area for prop in nuc_props])
    num_lobes = len(nuc_props)

    features = {
        'area_ratio': nuc_total_area/cell_props.area,
        'cell_eccentricity': cell_props.eccentricity,
        'num_lobes': num_lobes,
        ''
        'nucleus_circularity': (4 * np.pi * nuc_total_area) / (cell_props.perimeter ** 2) if cell_props.perimeter > 0 else 0
    }
    return features

features_list = []
labels_list = []

train_idxs=np.random.randint(0, len(train_files), len(train_files))
training_size = int(0.8*len(train_idxs))
print("Extracting features ----")
for train_idx in tqdm(train_idxs, desc="Extracting... "):
    X = io.imread(f"./train/{train_files[train_idx]}")
    white_blood_cell, nucleus_mask = multi_otsu_method(X)
    labelX = label_dict[train_files[train_idx]]
    feats = extract_features(white_blood_cell, nucleus_mask)
    features_list.append(feats)
    labels_list.append(labelX)

clf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
def train_and_validate(features_list, labels_list):
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    X_train, X_valid, y_train, y_valid=train_test_split(features_list, labels_list, test_size=0.2, random_state=0)
    df_features = pd.DataFrame(X_train)
    clf.fit(df_features, y_train)
    return clf, X_train, X_valid, y_train, y_valid

def validate(X_valid, clf):
    validation_features = pd.DataFrame(X_valid)
    y_pred=clf.predict(validation_features)
    return y_pred

def f1score(y_valid, y_pred):
    Conf = confusion_matrix(y_valid, y_pred)
    metrics = {}
    for k in range(len(Conf)):
        TP = Conf[k,k]
        FN = Conf[k,:].sum()-TP
        FP = Conf[:,k].sum()-TP
        TN = Conf.sum()-(TP+FN+FP)
        metrics[k]={
            "TP":TP,
            "FN":FN,
            "FP":FP,
            "TN":TN
        }

    precision=[metrics[k]["TP"]/(metrics[k]["TP"]+metrics[k]["FP"]) for k in range(len(Conf))]
    recall = [metrics[k]["TP"]/(metrics[k]["TP"]+metrics[k]["FN"]) for k in range(len(Conf))]

    F1score = [2*precision[k]*recall[k]/(precision[k]+recall[k]) for k in range(len(Conf))]
    return np.mean(F1score)

def train(features_list, labels_list):
    df_features = pd.DataFrame(features_list)
    clf.fit(df_features, labels_list)

def extract_test(test_files):
    features_list=[]
    for indx in tqdm(range(len(test_files))):
        X = io.imread(f"./test/{test_files[indx]}")
        white_blood_cell, nucleus_mask = multi_otsu_method(X)
        feats = extract_features(white_blood_cell, nucleus_mask)
        features_list.append(feats)
    return features_list

def test(features_test):
    y_pred = clf.predict(pd.DataFrame(features_test))
    test_df["label"] = y_pred
    test_df.to_csv('submission.csv', index=False)

train(features_list, labels_list)
features_test=extract_test(test_files)
test(features_test)
    