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

def random_parameters(data, K):
    m = (data.shape)[1]
    sigma = []
    mu = np.zeros((K, m))
    pi = np.ones(K)*1.0/K

    for k in range(K):
        idx = int(np.floor(np.random.random()*len(data)))
        sigma.append(np.cov(data.T))
        for col in range(m): mu[k][col] += data[idx][col]

    return mu, sigma, pi

def e_step(data, K, mu, sigma, pi):
    n = (data.shape)[0]
 
    resp = np.zeros((n, K))
 
    for i in range(n):
        den = 0
        for j in range(K):
            num = sp.multivariate_normal.pdf(data[i],mu[j],sigma[j],True)*pi[j]
            den += num
            resp[i][j]=num
        
        for j in range(K): resp[i][j]=resp[i][j]/den

    return resp

def m_step(data, K, resp):
    n = (data.shape)[0]
    m = (data.shape)[1]
    
    mu = np.zeros((K, m))
    sigma = np.zeros((K, m, m))
    pi = np.zeros(K)
    marg_resp = np.zeros(K)
    
    for k in range(K):
        for i in range(n):
            marg_resp[k] += resp[i][k]
            mu[k] += (resp[i][k])*data[i]
        mu[k] /= marg_resp[k]

        for i in range(n):
            x_mu = np.zeros((1,m))+data[i]-mu[k]
            sigma[k] += (resp[i][k])*x_mu*np.transpose(x_mu)

        sigma[k] /= marg_resp[k]
        pi[k] = marg_resp[k]/n
        
    return mu, sigma, pi


def EM(data,K,etiquetas):
    
    mu, sigma, pi = random_parameters(data, K)
    
    max_iter = 60
    logl=1
    epc_list=[];log_list=[]
    for it in range(max_iter):
        resp = e_step(data, K, mu, sigma, pi)
        mu, sigma, pi = m_step(data, K, resp)
        logl=log_likelihood(data,K,mu,sigma,pi)
        log_list.append(logl);epc_list.append(it)
        print(it,"log likelihood",logl)
    
    
    plt.plot(epc_list, log_list)
    plt.show()
    ans=assign_clusters(K,resp)
    print("Rand index",adjusted_rand_score(etiquetas,ans))
    
def log_likelihood(data,K,mu,sigma,pi):
    ans=0
    n=len(data)
    for i in range(n):
        temp=0
        for j in range(K):
            temp=temp+ sp.multivariate_normal.pdf(data[i],mu[j],sigma[j],True)*pi[j]
        ans=ans+np.log(temp)
    return ans


def assign_clusters(K, resp):
    idvs = len(resp)
    clusters = np.zeros(idvs, dtype=int)

    for i in range(idvs):
        clss = 0
        for k in range(K):
            if resp[i][k] > resp[i][clss]:
                clss = k
        clusters[i] = clss
    
    print("CLUSTERS")
    print(len(clusters))
    print(clusters)
    return clusters

def train(X,etiquetas):
    pca = PCA(n_components=78)
    principalComponents = pca.fit_transform(X)
    print("Dimensiones",len(principalComponents),len(principalComponents[0]))
    EM(principalComponents, 6,etiquetas)

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