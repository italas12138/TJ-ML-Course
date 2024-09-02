import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

###本文件使用库函数完成聚类，仅用于绘图，不用做实际实验过程

#csvstr = 'C:\\Users\\l\\Desktop\\PCA_3.csv'
csvstr = 'C:\\Users\\l\\Desktop\\PCA_2.csv'
#csvstr = 'C:\\Users\\l\\Desktop\\data_use.csv'

mydata = pd.read_csv(csvstr, dtype={'class': str})
mydata_array = mydata.values

# 删除第0列（分类标签）
data_for_clustering = np.delete(mydata_array, 0, axis=1)

linked = linkage(data_for_clustering, method='single', metric='euclidean')

plt.figure(figsize=(16, 7))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.show()

#clusters = fcluster(linked, 3, criterion='maxclust')
# 输出聚类结果
#print(clusters)
