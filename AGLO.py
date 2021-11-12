import numpy as np
import random as rd
import math
import csv
from numpy.random import normal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as sp
import matplotlib.pyplot as plt
from  sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_ ,counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def train(X,etiquetas):
    pca = PCA(0.95)
    principalComponents = pca.fit_transform(X)
    print("Dimensiones",len(principalComponents),len(principalComponents[0]))
    
    clustering = AgglomerativeClustering(n_clusters=6).fit(X)
    print(clustering.labels_)
    print("Rand index",adjusted_rand_score(etiquetas,clustering.labels_))

    #plt.title("Hierarchical Clustering Dendrogram")
    # plot the top three levels of the dendrogram
    #plot_dendrogram(clustering, truncate_mode="level", p=3)
    #plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    #plt.show()

def get_data(etiquetas):
    x=[]
    with open('dataset_tissue.csv', 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for i in csv_reader: 
            col=[]
            for j in range(1,len(i)): col.append(float(i[j]))
            x.append(np.array(col))

    print(len(x),len(x[0]))
    x=np.transpose(x)
    scaler=StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    print(len(x),len(x[0]))
        
    train(x,etiquetas)

def labels():
    x={'kidney': 0, 'hippocampus': 1, 'cerebellum': 2, 'colon': 3, 'liver': 4, 'endometrium': 5, 'placenta': 6} 
    etiquetas=[]
    with open('clase.csv', 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for i in csv_reader: 
            for j in range(1,len(i)): 
                etiquetas.append(x[i[j]])
    print("Labels")
    print(etiquetas)
    return etiquetas

def llamada():
    np.random.seed(0)
    etiquetas=labels()
    get_data(etiquetas)
    return

llamada()