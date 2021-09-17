from laspy.file import File
import numpy as np
import laspy as lp
import matplotlib.pyplot as plt
import copy
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import path

print("caricamento file...")
#input_path="C:/Users/crist/OneDrive/Documenti/Univpm/4° Anno/Computer Vision e Deep Learning/ProgettoUlivi/"
#dataname="prima_potatura_normalizzato_senzaterreno"

input_path1 = "C:/Users/crist/OneDrive/Documenti/Univpm/4° Anno/Computer Vision e Deep Learning/ProgettoUlivi/DatasetUliviPre1/"
dataname1 = "segmentazione10ulivipre"
point_cloud1 = lp.file.File(input_path1+dataname1+".las", mode="r")
print("File caricato.")

points = np.vstack((point_cloud1.x, point_cloud1.y, point_cloud1.z)).transpose()
#colors = np.vstack((point_cloud.red, point_cloud.green, point_cloud.blue)).transpose()
print("loading...")

factor = 20
decimated_points_random = points[::factor]

#decimated_colors = colors[::factor]
ax = plt.axes(projection='3d')
ax.scatter(decimated_points_random[:,0], decimated_points_random[:,1], decimated_points_random[:,2]) #c = decimated_colors/65535, s=0.01)
# plt.show()

print("Scrittura file di output...")
outFile1 = File("C:/Users/crist/OneDrive/Documenti/Univpm/4° Anno/Computer Vision e Deep Learning/ProgettoUlivi/outputpre.las", mode = "w", header = point_cloud1.header)
outFile1.points = point_cloud1.points[::20]
outFile1.close()
point_cloud1.close()
print("Fine.")

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("caricamento file...")
#input_path="C:/Users/crist/OneDrive/Documenti/Univpm/4° Anno/Computer Vision e Deep Learning/ProgettoUlivi/"
#dataname="prima_potatura_normalizzato_senzaterreno"

input_path2 = "C:/Users/crist/OneDrive/Documenti/Univpm/4° Anno/Computer Vision e Deep Learning/ProgettoUlivi/DatasetUliviPost1/"
dataname2 = "segmentazione10ulivipost"
point_cloud2 = lp.file.File(input_path2+dataname2+".las", mode="r")
print("File caricato.")

points = np.vstack((point_cloud2.x, point_cloud2.y, point_cloud2.z)).transpose()
#colors = np.vstack((point_cloud.red, point_cloud.green, point_cloud.blue)).transpose()
print("loading...")

factor = 20
decimated_points_random = points[::factor]

#decimated_colors = colors[::factor]
ax = plt.axes(projection='3d')
ax.scatter(decimated_points_random[:,0], decimated_points_random[:,1], decimated_points_random[:,2]) #c = decimated_colors/65535, s=0.01)
# plt.show()

print("Scrittura file di output...")
outFile2 = File("C:/Users/crist/OneDrive/Documenti/Univpm/4° Anno/Computer Vision e Deep Learning/ProgettoUlivi/outputpost.las", mode = "w", header = point_cloud2.header)
outFile2.points = point_cloud2.points[::20]
outFile2.close()
point_cloud2.close()
print("Fine.")

#print("Starting clustering...")
#clustering = DBSCAN(eps=2, min_samples=5, leaf_size=30).fit(decimated_points_random)

#core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
#core_samples_mask[clustering.core_sample_indices_] = True
#labels = clustering.labels_
# Number of clusters in labels, ignoring noise if present.
#n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#n_noise_ = list(labels).count(-1)
#print("Estimated number of clusters: %d " % n_clusters_)
#print("Estimated number of noise points: %d " % n_noise_)

# Black removed and is used for noise instead.
#fig = plt.figure(figsize=[100, 50])
#ax = fig.add_subplot(111, projection='3d')
#unique_labels = set(labels)
#colors = [plt.cm.Spectral(each)  for each in np.linspace(0, 1, len(unique_labels))]
#for k, col in zip(unique_labels, colors):
#    if k == -1:
#        col = [0, 0, 0, 1]
#    class_member_mask = (labels == k)
#    xyz = decimated_points_random[class_member_mask & core_samples_mask]
#    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=col, marker=".")
#plt.title("Estimated number of cluster: %d " % n_clusters_)
#plt.show()