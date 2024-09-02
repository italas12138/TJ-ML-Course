import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.metrics import fowlkes_mallows_score

#csvstr = 'C:\\Users\\l\\Desktop\\PCA_2.csv'
#csvstr = 'C:\\Users\\l\\Desktop\\PCA_3.csv'
csvstr = 'C:\\Users\\l\\Desktop\\data_use.csv'

#读取csv文件
mydata = pd.read_csv(csvstr, dtype={'class': str})
mydata_array = mydata.values

def divisive_clustering(data, num_clusters):
    clusters = [np.arange(data.shape[0])]
    cluster_to_split = clusters[0]
    kmeans = KMeans(n_clusters=num_clusters,init='k-means++').fit(data[cluster_to_split])
    new_clusters_labels = kmeans.labels_
    clusters=[]
    for it in range(num_clusters):
        clusters.append(cluster_to_split[new_clusters_labels == it])
    return clusters

# 删除第0列（分类标签）
data_for_clustering = np.delete(mydata_array, 0, axis=1)
# 执行分裂层次聚类
divisive_clusters = divisive_clustering(data_for_clustering, 3)
    
original_labels = mydata_array[:, 0]
cluster_labels = np.zeros_like(original_labels)
for cluster_id, cluster_indices in enumerate(divisive_clusters):
    for index in cluster_indices:
        cluster_labels[index] = cluster_id
fmi = fowlkes_mallows_score(original_labels, cluster_labels)

print(f"Fowlkes-Mallows Index: {fmi}")


