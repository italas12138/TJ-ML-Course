import pandas as pd
import numpy as np
from itertools import combinations

csvstr = 'C:\\Users\\l\\Desktop\\PCA_2.csv'
#csvstr = 'C:\\Users\\l\\Desktop\\PCA_3.csv'
#csvstr = 'C:\\Users\\l\\Desktop\\data_use.csv'

#读取csv文件
mydata = pd.read_csv(csvstr, dtype={'class': str})
mydata_array = mydata.values

# 删除第0列（分类标签）
data_for_clustering = np.delete(mydata_array, 0, axis=1)

# 计算欧式距离
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

# 计算曼哈顿距离
def manhattan_distance(point1, point2):
    return np.sum(np.abs(np.array(point1) - np.array(point2)))

# 计算切比雪夫距离
def chebyshev_distance(point1, point2):
    return np.max(np.abs(np.array(point1) - np.array(point2)))

# 便于调用
def calculate_distance(point1, point2):
    return euclidean_distance(point1, point2)
	#return manhattan_distance(point1, point2)
	#return chebyshev_distance(point1, point2)

# 单链聚类的距离计算
def single_linkage(cluster1, cluster2, data):
    min_distance = float('inf')
    for index1 in cluster1:
        for index2 in cluster2:
            distance = calculate_distance(data[index1], data[index2])
            if distance < min_distance:
                min_distance = distance
    return min_distance

# 完全链接聚类的距离计算
def complete_linkage(cluster1, cluster2, data):
    max_distance = float('-inf')
    for index1 in cluster1:
        for index2 in cluster2:
            distance = calculate_distance(data[index1], data[index2])
            if distance > max_distance:
                max_distance = distance
    return max_distance

# 组平均聚类的距离计算
def average_linkage(cluster1, cluster2, data):
    total_distance = 0
    num_pairs = 0
    for index1 in cluster1:
        for index2 in cluster2:
            distance = calculate_distance(data[index1], data[index2])
            total_distance += distance
            num_pairs += 1
    return total_distance / num_pairs if num_pairs > 0 else 0

# 计算聚类的中心点
def calculate_centroid(cluster_indices, data):
    points=[]
    for it in np.array(cluster_indices):
        points.append(data[it])
    centroid = np.mean(points, axis=0)
    return centroid
# 距离中心点聚类的距离计算
def centroid_linkage(cluster1, cluster2, data):
    # 计算每个聚类的中心点
    centroid1 = calculate_centroid(cluster1, data)
    centroid2 = calculate_centroid(cluster2, data)
    # 计算两个中心点之间的距离
    return calculate_distance(centroid1, centroid2)


# 层次聚类算法
def hierarchical_clustering(data, num_clusters):
    clusters = [[i] for i in range(len(data))]
    while len(clusters) > num_clusters:
        min_distance = float('inf')
        clusters_to_merge = (0, 1)
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                #几种不同的类间距离衡量
                #distance = single_linkage(clusters[i], clusters[j], data)      #MIN
                #distance = complete_linkage(clusters[i], clusters[j], data)    #MAX
                distance = average_linkage(clusters[i], clusters[j], data)     #Group Average
                #distance = centroid_linkage(clusters[i], clusters[j], data)    #Distance Between Centroids
                if distance < min_distance:
                    min_distance = distance
                    clusters_to_merge = (i, j)
        # 合并最近的两个簇
        clusters[clusters_to_merge[0]].extend(clusters[clusters_to_merge[1]])
        del clusters[clusters_to_merge[1]]
    return clusters

clusters = hierarchical_clustering(data_for_clustering.tolist(), 3)



# 计算Fowlkes-Mallows指数
original_labels = mydata_array[:, 0]
cluster_labels = np.zeros_like(original_labels)
for cluster_id, cluster in enumerate(clusters):
    for index in cluster:
        cluster_labels[index] = cluster_id
TP = FP = FN = TN = 0
for i, j in combinations(range(len(cluster_labels)), 2):
    same_cluster_original = (original_labels[i] == original_labels[j])
    same_cluster_new = (cluster_labels[i] == cluster_labels[j])
    if same_cluster_original and same_cluster_new:
        TP += 1
    elif not same_cluster_original and not same_cluster_new:
        TN += 1
    elif not same_cluster_original and same_cluster_new:
        FP += 1
    elif same_cluster_original and not same_cluster_new:
        FN += 1
FMI = TP / np.sqrt((TP + FP) * (TP + FN))


# 输出聚类结果
#for index, cluster in enumerate(clusters): print(f"Cluster {index + 1}: {cluster}")
# 输出Fowlkes-Mallows指数
print(f"Fowlkes-Mallows Index: {FMI}")