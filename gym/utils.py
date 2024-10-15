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
def save_args(args, file_path):
    with open(file_path, 'w') as f:
        json.dump(vars(args), f, indent=2,ensure_ascii=False)
class RetrievalData:
    def __init__(self,d,i,a,r=0) -> None:
        self.d=d
        self.i=i
        self.a=a
        self.r=r