import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray, rgb2hed
from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage.measure import regionprops
from skimage.morphology import dilation, erosion, disk, opening, closing, area_closing, area_opening
from skimage.measure import label
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

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

clf = RandomForestClassifier(n_estimators=100)
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
print("Extracting Done ----")
df_features = pd.DataFrame(features_list[:training_size])
print("Training")
clf.fit(df_features, labels_list[:training_size])
print("Training done")

validation_features = pd.DataFrame(features_list[training_size:])
score=clf.score(validation_features, labels_list[training_size:])
print(score)

