import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import json
def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum
def kmeans_filter(data,k=5):
    if data.shape[0]<k:
        mask=np.ones(shape=data.shape[0],dtype=bool)
        return mask
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    # 获取簇中心点的坐标
    centroids = kmeans.cluster_centers_
    # 获取每个样本所属的簇标签
    labels = kmeans.labels_
    cluster_sizes = Counter(labels)
    largest_cluster_label = max(cluster_sizes, key=cluster_sizes.get)
    #largest_cluster_size = cluster_sizes[largest_cluster_label]
    mask=(labels==largest_cluster_label)
    return mask
def save_args(args, file_path):
    with open(file_path, 'w') as f:
        json.dump(vars(args), f, indent=2,ensure_ascii=False)
class RetrievalData:
    def __init__(self,d,i,a,r=0) -> None:
        self.d=d
        self.i=i
        self.a=a
        self.r=r