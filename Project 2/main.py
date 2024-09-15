import cv2
from skimage.feature import hog
import numpy as np
import os
from tqdm import tqdm

def compute_hog(image, 
        hog_nbins = 9, 
        dim_hog_cell = 8, 
        two_dimensions = False, 
        gray = False, 
        blue = False, 
        green = True, 
        red = True
        ):
    # return list of hogs for chosen channels

    h = []

    if blue or green or red:
        b, g, r = cv2.split(image)

    if blue:
        h_b = hog(b, orientations=hog_nbins, pixels_per_cell = (dim_hog_cell, dim_hog_cell),
                        cells_per_block = (2, 2), feature_vector = False)
        h.append(h_b)
    if green:
        h_g = hog(g, orientations=hog_nbins, pixels_per_cell = (dim_hog_cell, dim_hog_cell),
                        cells_per_block = (2, 2), feature_vector = False)
        h.append(h_g)
    if red:
        h_r = hog(r, orientations=hog_nbins, pixels_per_cell = (dim_hog_cell, dim_hog_cell),
                        cells_per_block = (2, 2), feature_vector = False)
        h.append(h_r)

    if gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h_gray = hog(image, orientations=hog_nbins, pixels_per_cell = (dim_hog_cell, dim_hog_cell),
                        cells_per_block = (2, 2), feature_vector = False)
        h.append(h_gray)

    hogs = np.concatenate(h, axis=4)

    if two_dimensions == False:
        hogs = hogs.flatten()

    return hogs

types = ["train", "validation", "test"]

for type in types:

    root_folder_path = f'/mnt/d/University/PML/Project 2/Vegetable Images/{type}'

    paths = [
        os.path.join(foldername, filename)
        for foldername, subfolders, filenames in os.walk(root_folder_path)
        for filename in filenames
        if filename.lower().endswith(('.jpg'))
    ]

    hogs = []

    for path in tqdm(paths):

        img = cv2.imread(path)
        img = cv2.resize(img, (64, 64))
        hogs.append(compute_hog(img))

    hogs_s = np.array(hogs)
    np.save(f"/mnt/d/University/PML/Project 2/Hogs3/{type}_hogs_arrays.npy", hogs_s)

###################################
import numpy as np

# lod data
train_data = np.load('/mnt/d/University/PML/Project 2/Hogs3/train_hogs_arrays.npy')
test_data = np.load('/mnt/d/University/PML/Project 2/Hogs3/test_hogs_arrays.npy')
validation_data = np.load('/mnt/d/University/PML/Project 2/Hogs3/validation_hogs_arrays.npy')

#select only 2 classes
train_data = np.concatenate([train_data[2001:3001], train_data[7001:8001]], axis=0)
test_data = np.concatenate([test_data[401:601], test_data[1401:1601]], axis=0)
validation_data = np.concatenate([validation_data[401:601], validation_data[1401:1601]], axis=0)

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
##################################
#DBSCAN
#performe grid search
X = train_data

EPS = np.linspace(6, 7.5, 5)
MIN_SAMPLES = np.linspace(5, 500, 20)
UNBALANCE = np.linspace(0.55, 0.75, 5)
silhouettes = []

heatmap_matrix = -1 * np.ones((len(EPS), len(MIN_SAMPLES)))

for balance_treshold in UNBALANCE:
    best_silhouette = 0
    for i, eps in enumerate(EPS):
        for j, min_samples in enumerate(MIN_SAMPLES):
            min_samples = int(min_samples)
            dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan_model.fit_predict(X)
            uniq_labels, count_labels = np.unique(labels, return_counts=True)

            skip = False

            if len(uniq_labels) > 8:
                skip = True
                continue

            for no_cluster_samples in count_labels:
                if no_cluster_samples > balance_treshold * len(labels):
                    skip = True

            if skip:
                continue

            silhouette = silhouette_score(X, labels)
            heatmap_matrix[i, j] = silhouette

            if silhouette > best_silhouette:
                best_silhouette = silhouette


    # create a heatmap
    plt.figure(figsize=(16, 4))
    heatmap = plt.imshow(heatmap_matrix, cmap='viridis', origin='lower')

    # annotate rows and columns
    for i in range(len(EPS)):
        for j in range(len(MIN_SAMPLES)):
            value = heatmap_matrix[i, j]
            if value >= 0:  # use only positive values
                plt.text(j, i, f'{value:.3f}', ha='center', va='center', color='black', fontsize=8)

    # set axis labels and ticks
    plt.xticks(np.arange(len(MIN_SAMPLES)), [f'{s:.0f}' for s in MIN_SAMPLES], rotation=45)
    plt.yticks(np.arange(len(EPS)), [f'{eps:.2f}' for eps in EPS])

    plt.colorbar(label='Silhouette Score')
    plt.xlabel('Min Samples')
    plt.ylabel('Eps')
    plt.title(f'DBSCAN Hyperparameter Heatmap - Unbalanced threshold {balance_treshold:.2f}')

    plt.show()

    silhouettes.append(best_silhouette)
######################
# plot unblance vs score
plt.figure(figsize=(8, 6))
plt.plot(UNBALANCE, silhouettes, marker='o', linestyle='-', color='b')

plt.xlabel('Unbalance')
plt.ylabel('Silhouettes')
plt.title('Unbalance vs Silhouettes')

plt.grid(True)
plt.show()
################################
# DBSCAN metrics on test for ebst models on train clustering score
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

X = train_data
X_test = test_data

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)
gt_labels = [0 for _ in range(200)] + [1 for _ in range(200)]
plt.figure()

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gt_labels, cmap='viridis', s=10)
plt.title('GT Labels (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.show()

EPS = [7.12, 7.12, 7.12, 7.5, 7.5]
MIN_SAMPLES = [161, 83, 57, 396, 306]
UNBALANCE = [0.55, 0.6, 0.65, 0.7, 0.75]
silhouettes = []

for balance_treshold, eps, min_samples in zip(UNBALANCE, EPS, MIN_SAMPLES):

    min_samples = int(min_samples)
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan_model.fit_predict(X)
    uniq_labels, count_labels = np.unique(labels, return_counts=True)
    silhouette = silhouette_score(X, labels)

    knn_classifier = KNeighborsClassifier(n_neighbors=1)
    knn_classifier.fit(X, labels)
    test_labels = knn_classifier.predict(X_test)
    test_labels = [l if l==0 else 1 for l in test_labels]

    plt.figure()

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=test_labels, cmap='viridis', s=10)
    plt.title('DBSCAN Clustering (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.show()

    print(f"Eps: {eps} Min: {min_samples} Treshold {balance_treshold} Labels: {uniq_labels, count_labels}")
    print("Silhouette Score:", silhouette)
    print(classification_report(gt_labels, test_labels))

    conf_matrix = confusion_matrix(gt_labels, test_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=set(test_labels), yticklabels=set(test_labels))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
##############################################
# plto distances between shuffled data to find potential epsilon values
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import copy

shuffled_data = copy.deepcopy(train_data)
np.random.shuffle(shuffled_data)

# calculate Euclidean distances
distances_1 = cdist(shuffled_data, train_data, 'euclidean')[0]

plt.plot(distances_1, marker='o', label='Distances', alpha=0.5)
plt.title('Distances between samples')
plt.xlabel('Samples')
plt.ylabel('Euclidean Distance')
plt.legend()
plt.show()
################################
# lod rgb histograms
import numpy as np

train_data = np.load('/mnt/d/University/PML/Project 2/Hists/train_hist_arrays.npy')
test_data = np.load('/mnt/d/University/PML/Project 2/Hists/test_hist_arrays.npy')
validation_data = np.load('/mnt/d/University/PML/Project 2/Hists/validation_hist_arrays.npy')

train_data = train_data.reshape(train_data.shape[0], -1)
test_data = test_data.reshape(test_data.shape[0], -1)
validation_data = validation_data.reshape(validation_data.shape[0], -1)
train_data = np.concatenate([train_data[2001:3001], train_data[7001:8001]], axis=0)
test_data = np.concatenate([test_data[401:601], test_data[1401:1601]], axis=0)
validation_data = np.concatenate([validation_data[401:601], validation_data[1401:1601]], axis=0)
print(train_data.shape)

###################################
# load vq-vae features:
import numpy as np

# Load the data
# train_t_data = np.load('/mnt/d/University/PML/Project 2/Encoder-Tiny/train_t_array.npy')
# test_t_data = np.load('/kaggle/input/pml-dataset-encoder-features/test_t_array.npy')
# validation_t_data = np.load('/kaggle/input/pml-dataset-encoder-features/validation_t_array.npy')
train_b_data = np.load('/mnt/d/University/PML/Project 2/Encoder-Tiny/train_b_array.npy')
test_b_data = np.load('/mnt/d/University/PML/Project 2/Encoder-Tiny/test_b_array.npy')
validation_b_data = np.load('/mnt/d/University/PML/Project 2/Encoder-Tiny/validation_b_array.npy')

train_data = train_b_data.reshape(train_b_data.shape[0], -1)
test_data = test_b_data.reshape(test_b_data.shape[0], -1)
validation_data = validation_b_data.reshape(validation_b_data.shape[0], -1)
train_data = np.concatenate([train_data[2001:3001], train_data[7001:8001]], axis=0)
test_data = np.concatenate([test_data[401:601], test_data[1401:1601]], axis=0)
validation_data = np.concatenate([validation_data[401:601], validation_data[1401:1601]], axis=0)
print(train_data.shape)

###################################
# Dummy classifyer

from sklearn.dummy import DummyClassifier
test_labels = np.concatenate([np.zeros(200), np.ones(200)])
for strategy in ["most_frequent", "prior", "stratified", "uniform"]:
    dummy_clf = DummyClassifier(strategy=strategy)
    dummy_clf.fit(test_data, test_labels)
    dummy_clf.predict(test_data)
    print(f"Strategy: {strategy}\nScore: "+str(dummy_clf.score(test_data, test_labels)))

#####################################
# SVM
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

train_labels = np.concatenate([np.zeros(1000), np.ones(1000)])
val_labels = np.concatenate([np.zeros(200), np.ones(200)])
test_labels = np.concatenate([np.zeros(200), np.ones(200)])

best_score = 0
best_c = 0
for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    svm = SVC(C=c)
    svm.fit(train_data, train_labels)
    labels = svm.predict(validation_data)

    score = accuracy_score(labels, val_labels)

    if score > best_score:
        best_score = score
        best_c = c

best_svm = SVC(C=best_c)
best_svm.fit(train_data, train_labels)

test_predictions = best_svm.predict(test_data)
print(f"Best validation accuracy: {best_score} with C={best_c}")
print("Classification Report for the Best SVM Model on Test Set:")
print(classification_report(test_labels, test_predictions))

conf_matrix = confusion_matrix(test_labels, test_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=set(labels), yticklabels=set(labels))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
############################################
# Random forest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

train_labels = np.concatenate([np.zeros(1000), np.ones(1000)])
val_labels = np.concatenate([np.zeros(200), np.ones(200)])
test_labels = np.concatenate([np.zeros(200), np.ones(200)])

best_score = 0
best_n_estimators = 0

for n_estimators in [50, 100, 200, 300]:  
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_classifier.fit(train_data, train_labels)
    labels = rf_classifier.predict(validation_data)

    score = accuracy_score(labels, val_labels)

    if score > best_score:
        best_score = score
        best_n_estimators = n_estimators

best_rf = RandomForestClassifier(n_estimators=best_n_estimators, random_state=42)
best_rf.fit(train_data, train_labels)

test_predictions = best_rf.predict(test_data)
print(f"Best validation accuracy: {best_score} with n_estimators={best_n_estimators}")
print("Classification Report for the Best Random Forest Model on Test Set:")
print(classification_report(test_labels, test_predictions))

conf_matrix = confusion_matrix(test_labels, test_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=set(labels), yticklabels=set(labels))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
################################
# GMM

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

X = train_data

# params for GMM
params = {
    'n_components': 2,
    'covariance_type': 'full',
    'init_params': 'random_from_data',
    # 'random_state': 49
    # 44, 48 35000, 260, 20, 0-bad, 123-bad, 43, 45-bad, 46-bad, 47-bad, 49-bad
}

UNBALANCE = [0.55, 0.6, 0.65, 0.7, 0.75]
silhouettes = []


for balance_treshold in UNBALANCE:

    best_silhouette = 0

    for rand in np.linspace(1, 100, 100): # search for best seed
        

        gmm_model = GaussianMixture(random_state=int(rand), **params)
        labels = gmm_model.fit_predict(X)
        uniq_labels, count_labels = np.unique(labels, return_counts=True)

        skip = False

        for no_cluster_samples in count_labels:
            if no_cluster_samples > balance_treshold * len(labels): #unbalance class condition
                skip = True

        if skip:
            continue

        silhouette = silhouette_score(X, labels)

        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_seed = int(rand)
    
    print(best_silhouette)

    silhouettes.append((best_silhouette, best_seed))

print(silhouettes)

# plot seed vs score
# values were manualy taken form previously printed pairs
plt.plot([0.55, 0.6, 0.65, 0.7, 0.75],
[0.05417870511483591,
0.061942464479300685,
0.06458484085145505,
0.06458484085145505,
0.06469657837768596], 'o-')

plt.title('GMM - HOG')
plt.xlabel('Blance threshold')
plt.ylabel('Silhouette')
plt.legend()
plt.grid(True)
plt.show()

###############################
# grid search on tolerance
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


X = train_data

params = {
    'n_components': 2,
    'covariance_type': 'full',
    'init_params': 'random_from_data',

    # 44, 48 35000, 260, 20, 0-bad, 123-bad, 43, 45-bad, 46-bad, 47-bad, 49-bad
}
UNBALANCE = [0.55, 0.6, 0.65, 0.7, 0.75]
SEEDS = [41, 21, 73, 73, 55]
TOL = [1e-3, 1e-6, 1e-9]
MAX_ITER = [100]
silhouettes = []

heatmap_matrix = -1 * np.ones((len(MAX_ITER), len(TOL)))

for balance_treshold, seed in zip(UNBALANCE, SEEDS):
    best_silhouette = 0
    for i, max_iter in enumerate(MAX_ITER):
        for j, tol in enumerate(TOL):
            gmm_model = GaussianMixture(tol=tol, max_iter=max_iter, random_state=seed, **params)
            labels = gmm_model.fit_predict(X)
            uniq_labels, count_labels = np.unique(labels, return_counts=True)

            skip = False

            for no_cluster_samples in count_labels:
                if no_cluster_samples > balance_treshold * len(labels):
                    skip = True

            if skip:
                continue

            silhouette = silhouette_score(X, labels)
            heatmap_matrix[i, j] = silhouette

            if silhouette > best_silhouette:
                best_silhouette = silhouette
    print(best_silhouette)

    plt.figure(figsize=(6, 4))
    heatmap = plt.imshow(heatmap_matrix, cmap='viridis', origin='lower')

    for i in range(len(MAX_ITER)):
        for j in range(len(TOL)):
            value = heatmap_matrix[i, j]
            if value >= 0:  
                plt.text(j, i, f'{value:.3f}', ha='center', va='center', color='black', fontsize=8)


    plt.xticks(np.arange(len(TOL)), [f'{t:.0e}' for t in TOL], rotation=45)
    plt.yticks(np.arange(len(MAX_ITER)), [f'{max_iter:.0f}' for max_iter in MAX_ITER])
    plt.colorbar(label='Silhouette Score')
    plt.ylabel('Max iterations')
    plt.xlabel('Tolerance')
    plt.title(f'GMM - Unbalanced threshold {balance_treshold} - Seed {seed}')
    plt.show()

    silhouettes.append(best_silhouette)

#############################################
# GMM on test data
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns

X = train_data
X_test = test_data
gt_labels = [0 for _ in range(200)] + [1 for _ in range(200)]
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)
plt.figure()

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gt_labels, cmap='viridis', s=5, alpha=0.33)
plt.title('GT Labels (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.show()

params = {
    'n_components': 2,
    'covariance_type': 'full',
    'init_params': 'random_from_data',

    # 44, 48 35000, 260, 20, 0-bad, 123-bad, 43, 45-bad, 46-bad, 47-bad, 49-bad
}
UNBALANCE = [0.55, 0.6, 0.65, 0.7, 0.75]
SEEDS = [41, 21, 73, 73, 55]
TOL = [1e-3, 1e-6, 1e-9]
MAX_ITER = [100]
silhouettes = []

heatmap_matrix = -1 * np.ones((len(MAX_ITER), len(TOL)))

for balance_treshold, seed in zip(UNBALANCE, SEEDS):
    best_silhouette = 0

    gmm_model = GaussianMixture(random_state=seed, **params)
    gmm_model.fit(X)
    labels = gmm_model.predict(X_test)
    if accuracy_score(gt_labels, labels) < 0.5:
        labels = [1 if l==0 else 0 for l in labels]
    plt.figure()

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=5, alpha=0.33)
    plt.title(f'GMM Clustering - Unbalance threshold {balance_treshold} - Seed {seed} (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.show()


    print(classification_report(gt_labels, labels))

    conf_matrix = confusion_matrix(gt_labels, labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=set(labels), yticklabels=set(labels))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
#########################################
# histogram extraction
import cv2
from skimage.feature import hog
import numpy as np
import os
from tqdm import tqdm

def compute_hist(image):
    h = []

    b, g, r = cv2.split(image)

    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

    h.append(hist_b)
    h.append(hist_g)
    h.append(hist_r)

    return h


types = ["train", "validation", "test"]

for type in types:

    root_folder_path = f'/mnt/d/University/PML/Project 2/Vegetable Images/{type}'

    paths = [
        os.path.join(foldername, filename)
        for foldername, subfolders, filenames in os.walk(root_folder_path)
        for filename in filenames
        if filename.lower().endswith(('.jpg'))
    ]

    hists = []

    for path in tqdm(paths):

        img = cv2.imread(path)
        img = cv2.resize(img, (224, 224))
        hists.append(compute_hist(img))

    hists_s = np.array(hists)
    np.save(f"/mnt/d/University/PML/Project 2/Hists/{type}_hist_arrays.npy", hists_s)
#############################################
# vae model
import torch
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F

#https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f


# Load the data
train_t_data = np.load('/mnt/d/University/PML/Project 2/Encoder-Tiny/train_t_array.npy')
train_data = train_t_data.reshape(train_t_data.shape[0], -1)
validation_t_data = np.load('/mnt/d/University/PML/Project 2/Encoder-Tiny/validation_t_array.npy')
validation_t_data = validation_t_data.reshape(validation_t_data.shape[0], -1)


# Combine both train and validation data
combined_data = np.vstack([train_data, validation_t_data])

# Initialize the scaler
scaler = MinMaxScaler()

# Fit the scaler on the combined data and transform both training and validation data
combined_data_scaled = scaler.fit_transform(combined_data)
train_data_scaled = combined_data_scaled[:len(train_data)]
validation_data_scaled = combined_data_scaled[len(train_data):]

# Convert numpy arrays to PyTorch tensors
train_data_scaled_tensor = torch.tensor(train_data_scaled, dtype=torch.float32)
validation_data_scaled_tensor = torch.tensor(validation_data_scaled, dtype=torch.float32)

# Create TensorDatasets
train_dataset = TensorDataset(train_data_scaled_tensor)
validation_dataset = TensorDataset(validation_data_scaled_tensor)

# Create train and validation dataloaders
batch_size = 1000
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE(nn.Module):

    def __init__(self, input_dim=196, hidden_dim=128, latent_dim=64, device=device):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.SiLU(),
            nn.BatchNorm1d(latent_dim),
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(latent_dim, 1)
        self.logvar_layer = nn.Linear(latent_dim, 1)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.BatchNorm1d(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
            )
     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)      
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar
        

model = VAE().to(device)
optimizer = AdamW(model.parameters(), lr=2e-2)

def loss_function(x, x_hat, mean, log_var):
    # Flatten x to match the shape of x_hat
    x = x.view(-1, 196)

    reproduction_loss = F.mse_loss(x_hat, x)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

def train(model, optimizer, epochs, device, train_loader, validation_loader, model_save_path):
    best_loss = float('inf')

    # Lists to store compressed representations
    train_compressed = []
    validation_compressed = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_idx, x in enumerate(train_loader):
            x = x[0].to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            # Append compressed representations to the list
            if epoch == 0:
                train_compressed.append((mean.cpu().detach().numpy(), log_var.cpu().detach().numpy()))

        # Validation
        model.eval()
        validation_loss = 0

        with torch.no_grad():
            for batch_idx, x_val in enumerate(validation_loader):
                x_val = x_val[0].to(device)

                x_hat_val, mean_val, log_var_val = model(x_val)
                loss_val = loss_function(x_val, x_hat_val, mean_val, log_var_val)

                validation_loss += loss_val.item()

                # Append compressed representations to the list
                if epoch == 0:
                    validation_compressed.append((mean_val.cpu().detach().numpy(), log_var_val.cpu().detach().numpy()))

        # Average losses
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_validation_loss = validation_loss / len(validation_loader.dataset)

        print(f"Epoch {epoch + 1}/{epochs} \t Train Loss: {avg_train_loss:.4f} \t Validation Loss: {avg_validation_loss:.6f}")

        # Save the model if it has the best validation loss
        if avg_validation_loss < best_loss:
            best_loss = avg_validation_loss
            torch.save(model.state_dict(), model_save_path)
            print("Model saved!")

    # Convert lists to numpy arrays and save
    train_compressed_np = np.array(train_compressed).reshape(-1, 2)
    validation_compressed_np = np.array(validation_compressed).reshape(-1, 2)

    print(validation_compressed_np.shape)

    np.save('/mnt/d/University/PML/Project 2/Encoder-TIny-VAE/train_compressed.npy', train_compressed_np)
    np.save('/mnt/d/University/PML/Project 2/Encoder-TIny-VAE/validation_compressed.npy', validation_compressed_np)

    return best_loss



model_save_path = '/mnt/d/University/PML/Project 2/vae/best_model.pth'

# Call the training function
best_loss = train(model, optimizer, epochs=50, device=device, train_loader=train_loader,
                  validation_loader=validation_loader, model_save_path=model_save_path)

print(f"Best Validation Loss: {best_loss:.6f}")

#########################################3
# vq-vae
from math import cos, pi, floor, sin

from torch.optim import lr_scheduler


class CosineLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_min, lr_max, step_size):
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.step_size = step_size
        self.iteration = 0

        super().__init__(optimizer, -1)

    def get_lr(self):
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + cos(self.iteration / self.step_size * pi)
        )
        self.iteration += 1

        if self.iteration == self.step_size:
            self.iteration = 0

        return [lr for base_lr in self.base_lrs]


class PowerLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_min, lr_max, warmup):
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warmup = warmup
        self.iteration = 0

        super().__init__(optimizer, -1)

    def get_lr(self):
        if self.iteration < self.warmup:
            lr = (
                self.lr_min + (self.lr_max - self.lr_min) / self.warmup * self.iteration
            )

        else:
            lr = self.lr_max * (self.iteration - self.warmup + 1) ** -0.5

        self.iteration += 1

        return [lr for base_lr in self.base_lrs]


class SineLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_min, lr_max, step_size):
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.step_size = step_size
        self.iteration = 0

        super().__init__(optimizer, -1)

    def get_lr(self):
        lr = self.lr_min + (self.lr_max - self.lr_min) * sin(
            self.iteration / self.step_size * pi
        )
        self.iteration += 1

        if self.iteration == self.step_size:
            self.iteration = 0

        return [lr for base_lr in self.base_lrs]


class LinearLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_min, lr_max, warmup, step_size):
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.step_size = step_size
        self.warmup = warmup
        self.iteration = 0

        super().__init__(optimizer, -1)

    def get_lr(self):
        if self.iteration < self.warmup:
            lr = self.lr_max

        else:
            lr = self.lr_max + (self.iteration - self.warmup) * (
                self.lr_min - self.lr_max
            ) / (self.step_size - self.warmup)
        self.iteration += 1

        if self.iteration == self.step_size:
            self.iteration = 0

        return [lr for base_lr in self.base_lrs]


class CLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_min, lr_max, step_size):
        self.epoch = 0
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.current_lr = lr_min
        self.step_size = step_size

        super().__init__(optimizer, -1)

    def get_lr(self):
        cycle = floor(1 + self.epoch / (2 * self.step_size))
        x = abs(self.epoch / self.step_size - 2 * cycle + 1)
        lr = self.lr_min + (self.lr_max - self.lr_min) * max(0, 1 - x)
        self.current_lr = lr

        self.epoch += 1

        return [lr for base_lr in self.base_lrs]


class Warmup(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, model_dim, factor=1, warmup=16000):
        self.optimizer = optimizer
        self.model_dim = model_dim
        self.factor = factor
        self.warmup = warmup
        self.iteration = 0

        super().__init__(optimizer, -1)

    def get_lr(self):
        self.iteration += 1
        lr = (
            self.factor
            * self.model_dim ** (-0.5)
            * min(self.iteration ** (-0.5), self.iteration * self.warmup ** (-1.5))
        )

        return [lr for base_lr in self.base_lrs]


# Copyright 2019 fastai

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Borrowed from https://github.com/fastai/fastai and changed to make it runs like PyTorch lr scheduler


class CycleAnnealScheduler:
    def __init__(
        self, optimizer, lr_max, lr_divider, cut_point, step_size, momentum=None
    ):
        self.lr_max = lr_max
        self.lr_divider = lr_divider
        self.cut_point = step_size // cut_point
        self.step_size = step_size
        self.iteration = 0
        self.cycle_step = int(step_size * (1 - cut_point / 100) / 2)
        self.momentum = momentum
        self.optimizer = optimizer

    def get_lr(self):
        if self.iteration > 2 * self.cycle_step:
            cut = (self.iteration - 2 * self.cycle_step) / (
                self.step_size - 2 * self.cycle_step
            )
            lr = self.lr_max * (1 + (cut * (1 - 100) / 100)) / self.lr_divider

        elif self.iteration > self.cycle_step:
            cut = 1 - (self.iteration - self.cycle_step) / self.cycle_step
            lr = self.lr_max * (1 + cut * (self.lr_divider - 1)) / self.lr_divider

        else:
            cut = self.iteration / self.cycle_step
            lr = self.lr_max * (1 + cut * (self.lr_divider - 1)) / self.lr_divider

        return lr

    def get_momentum(self):
        if self.iteration > 2 * self.cycle_step:
            momentum = self.momentum[0]

        elif self.iteration > self.cycle_step:
            cut = 1 - (self.iteration - self.cycle_step) / self.cycle_step
            momentum = self.momentum[0] + cut * (self.momentum[1] - self.momentum[0])

        else:
            cut = self.iteration / self.cycle_step
            momentum = self.momentum[0] + cut * (self.momentum[1] - self.momentum[0])

        return momentum

    def step(self):
        lr = self.get_lr()

        if self.momentum is not None:
            momentum = self.get_momentum()

        self.iteration += 1

        if self.iteration == self.step_size:
            self.iteration = 0

        for group in self.optimizer.param_groups:
            group['lr'] = lr

            if self.momentum is not None:
                group['betas'] = (momentum, group['betas'][1])

        return lr


def anneal_linear(start, end, proportion):
    return start + proportion * (end - start)


def anneal_cos(start, end, proportion):
    cos_val = cos(pi * proportion) + 1

    return end + (start - end) / 2 * cos_val


class Phase:
    def __init__(self, start, end, n_iter, anneal_fn):
        self.start, self.end = start, end
        self.n_iter = n_iter
        self.anneal_fn = anneal_fn
        self.n = 0

    def step(self):
        self.n += 1

        return self.anneal_fn(self.start, self.end, self.n / self.n_iter)

    def reset(self):
        self.n = 0

    @property
    def is_done(self):
        return self.n >= self.n_iter


class CycleScheduler:
    def __init__(
        self,
        optimizer,
        lr_max,
        n_iter,
        momentum=(0.95, 0.85),
        divider=25,
        warmup_proportion=0.3,
        phase=('linear', 'cos'),
    ):
        self.optimizer = optimizer

        phase1 = int(n_iter * warmup_proportion)
        phase2 = n_iter - phase1
        lr_min = lr_max / divider

        phase_map = {'linear': anneal_linear, 'cos': anneal_cos}

        self.lr_phase = [
            Phase(lr_min, lr_max, phase1, phase_map[phase[0]]),
            Phase(lr_max, lr_min / 1e4, phase2, phase_map[phase[1]]),
        ]

        self.momentum = momentum

        if momentum is not None:
            mom1, mom2 = momentum
            self.momentum_phase = [
                Phase(mom1, mom2, phase1, phase_map[phase[0]]),
                Phase(mom2, mom1, phase2, phase_map[phase[1]]),
            ]

        else:
            self.momentum_phase = []

        self.phase = 0

    def step(self):
        lr = self.lr_phase[self.phase].step()

        if self.momentum is not None:
            momentum = self.momentum_phase[self.phase].step()

        else:
            momentum = None

        for group in self.optimizer.param_groups:
            group['lr'] = lr

            if self.momentum is not None:
                if 'betas' in group:
                    group['betas'] = (momentum, group['betas'][1])

                else:
                    group['momentum'] = momentum

        if self.lr_phase[self.phase].is_done:
            self.phase += 1

        if self.phase >= len(self.lr_phase):
            for phase in self.lr_phase:
                phase.reset()

            for phase in self.momentum_phase:
                phase.reset()

            self.phase = 0

        return lr, momentum


class LRFinder(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_min, lr_max, step_size, linear=False):
        ratio = lr_max / lr_min
        self.linear = linear
        self.lr_min = lr_min
        self.lr_mult = (ratio / step_size) if linear else ratio ** (1 / step_size)
        self.iteration = 0
        self.lrs = []
        self.losses = []

        super().__init__(optimizer, -1)

    def get_lr(self):
        lr = (
            self.lr_mult * self.iteration
            if self.linear
            else self.lr_mult ** self.iteration
        )
        lr = self.lr_min + lr if self.linear else self.lr_min * lr

        self.iteration += 1
        self.lrs.append(lr)

        return [lr for base_lr in self.base_lrs]

    def record(self, loss):
        self.losses.append(loss)

    def save(self, filename):
        with open(filename, 'w') as f:
            for lr, loss in zip(self.lrs, self.losses):
                f.write('{},{}\n'.format(lr, loss))
##############################################################
import torch
from torch import nn
from torch.nn import functional as F


# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out

# class HeadAttention(nn.Module):
#     def __init__(self, channels, size):
#         super(HeadAttention, self).__init__()
#         self.channels = channels
#         self.size = size
#         self.mha = nn.MultiheadAttention(channels, 3, batch_first=True)
#         self.ln = nn.LayerNorm([channels])
#         self.ff_self = nn.Sequential(
#             nn.LayerNorm([channels]),
#             nn.Linear(channels, channels),
#             nn.GELU(),
#             nn.Linear(channels, channels),
#         )

#     def forward(self, x):
#         x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
#         x_ln = self.ln(x)
#         attention_value, _ = self.mha(x_ln, x_ln, x_ln)
#         attention_value = attention_value + x
#         attention_value = self.ff_self(attention_value) + attention_value
#         return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 6:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))
        
        # if stride == 4:
        #     blocks.append(HeadAttention(channel, 64))
        # elif stride == 2:
        #     blocks.append(HeadAttention(channel, 32))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))
    
        # if stride == 4:
        #     blocks.append(HeadAttention(channel, 64))
        # elif stride == 2:
        #     blocks.append(HeadAttention(channel, 32))

        if stride == 6:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(channel // 2, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=1,
        n_embed=1024,
        decay=0.99,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=6)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1)
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=6,
        )
        self.activation = nn.Tanh()

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return self.activation(dec), diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec
#######################################################
import argparse
import os
import sys

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils
from torch.utils import data
from tqdm import tqdm

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


def train(epoch, loader, val_loader, model, optimizer, scheduler, device):

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25

    train_losses, val_losses = 0.0, 0.0

    
    for i, (img, _) in enumerate(loader):
        model.zero_grad()
        
        epoch_loss = 0
        total_inputs = 0

        img = img.to(device)

#         out, latent_loss = model(img + torch.randn_like(img))
        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()
        
        epoch_loss += loss.item()*len(img)

        total_inputs += len(img)


        if i%100 == 0:
            print("Train loss: ", loss.item())
            print("MSE loss: ", recon_loss.item())
            print()


    train_losses = (epoch_loss / total_inputs)

    epoch_loss = 0
    total_inputs = 0

    with torch.no_grad():

        for data in val_loader:

            inputs, _ = data
            inputs = inputs.to(device)

#             outputs, latent_loss = model(inputs + torch.randn_like(img))
            outputs, latent_loss = model(inputs)
            mse = criterion(outputs, inputs)
            loss = mse + latent_loss * latent_loss_weight


            epoch_loss += loss.item()*len(inputs)
            total_inputs += len(inputs)

        val_losses = (epoch_loss / total_inputs)

    print("Epoch {}:".format(epoch))
    print("Train loss: ", train_losses)
    print("Validation loss: ", val_losses)

    model.train()

    return train_losses, val_losses


def main(args):
    
    device = "cuda"

    train_losses, val_losses = [], []

    transform = transforms.Compose([
        transforms.Resize((args["size"], args["size"])),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(args["path_train"], transform=transform)
    loader = DataLoader(dataset, batch_size=args["batch"], shuffle=True)
    
    dataset_val = datasets.ImageFolder(args["path_val"], transform=transform)
    loader_val = DataLoader(dataset, batch_size=args["batch"], shuffle=False)

    model = VQVAE().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args["lr"])
    scheduler = None
    if args["sched"] == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args["lr"],
            n_iter=len(loader) * args["epoch"],
            momentum=None,
            warmup_proportion=0.15,
        )

    start_epoch = 0
    #################################################
    # load checkpoint
#     checkpoint = torch.load('/kaggle/input/pml-project-2-models/vqvae_017.pth')
#     model.load_state_dict(checkpoint['model_state_dict'])
#     del checkpoint
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     with open("G3/vqvae_sch_2.pkl", "rb") as file:
#       schedduler = pickle.load(file)
#     start_epoch = checkpoint['epoch']
    ###################################################
    
    for i in range(start_epoch + 1, args["epoch"] + 1):
        t_loss, v_loss = train(i, loader, loader_val, model, optimizer, scheduler, device)

        train_losses.append(t_loss)
        val_losses.append(v_loss)


        torch.save({
          'epoch': i,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'train_loss': train_losses,
          'val_loss': val_losses,
        }, f"vqvae_small_{str(i).zfill(3)}.pth")
        with open(f"vqvae_sch_{str(i)}.pkl", "wb") as file:
            pickle.dump(scheduler, file, -1)


config = {
    "batch": 64,
    "size": 224,
    "epoch": 20,
    "lr": 15e-4,
    "sched": 'cycle',
    "path_train": '/kaggle/input/vegetable-image-dataset/Vegetable Images/train',
    "path_val": '/kaggle/input/vegetable-image-dataset/Vegetable Images/validation'
}

print(config)

main(config)
######################################
#vq-vae inference

import argparse
import os
import sys

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils
from torch.utils import data
from tqdm import tqdm

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
import numpy as np

def inference(loader, model, device):
    
    features_t = []
    features_b = []

    with torch.no_grad():
        for data in loader:
            inputs, _ = data
            inputs = inputs.to(device)
            quant_t, quant_b, _, _, _ = model.encode(inputs)
            print(quant_t.shape, quant_b.shape)
            features_t.append(quant_t.detach().cpu())
            features_b.append(quant_b.detach().cpu())
    
    return features_t, features_b


def main(args):
    
    device = "cuda"

    transform = transforms.Compose([
        transforms.Resize((args["size"], args["size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    path = f"/kaggle/input/vegetable-image-dataset/Vegetable Images/{args['type']}"
    
    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, batch_size=args["batch"], shuffle=False)

    model = VQVAE().to(device)

    checkpoint = torch.load('/kaggle/input/pml-project-2-models/vqvae_small_016.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint

    return inference(loader, model, device)


types = ["train", "validation", "test"]
for type in types:

    config = {
        "batch": 100,
        "type": type,
        "size": 224,
        "path_train": '/kaggle/input/vegetable-image-dataset/Vegetable Images/train',
        "path_val": '/kaggle/input/vegetable-image-dataset/Vegetable Images/validation'
    }

    print(config)

    features_t, features_b = main(config)

    features_t_np = np.array([t.numpy() for t in features_t])
    print(features_t_np.shape)
    features_t_np = features_t_np.reshape(features_t_np.shape[0]*features_t_np.shape[1], features_t_np.shape[2], features_t_np.shape[3], features_t_np.shape[4])
    features_b_np = np.array([b.numpy() for b in features_b])
    features_b_np = features_b_np.reshape(features_b_np.shape[0]*features_b_np.shape[1], features_b_np.shape[2], features_b_np.shape[3], features_b_np.shape[4])


    np.save(f'{type}_t_array.npy', features_t_np)
    np.save(f'{type}_b_array.npy', features_b_np)
    
    print(f"saved {type}")