from laspy.file import File
import scipy
import laspy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import path
 
#import dataset pre pruning
inFilePre = File('C:/Users/crist/OneDrive/Documenti/Univpm/4° Anno/Computer Vision e Deep Learning/ProgettoUlivi/outputpre.las', mode='r')
datasetpre = np.vstack([inFilePre.x, inFilePre.y, inFilePre.z]).transpose()
datasetpre.shape


#import dataset post pruning
inFilePost = File('C:/Users/crist/OneDrive/Documenti/Univpm/4° Anno/Computer Vision e Deep Learning/ProgettoUlivi/outputpost.las', mode='r')
datasetpost = np.vstack([inFilePost.x, inFilePost.y, inFilePost.z]).transpose()
datasetpost.shape

#import dataset for header
input_path1 = "C:/Users/crist/OneDrive/Documenti/Univpm/4° Anno/Computer Vision e Deep Learning/ProgettoUlivi/DatasetUliviPre1/"
dataname1 = "segmentazione10ulivipre"
point_cloud1 = laspy.file.File(input_path1+dataname1+".las", mode="r")

def frange(start, stop, step):
  i = start
  while i < stop:
    yield i
    i += step
#ground points grid filter
#n = 100 #grid step
#dataset_Z_filtered = dataset[[0]]
#zfiltered = (dataset[:, 2].max() - dataset[:, 2].min())/10 #setting height filtered from ground
#print("zfiltered =", zfiltered)
#xstep = (dataset[:, 0].max() - dataset[:, 0].min())/n
#ystep = (dataset[:, 1].max() - dataset[:, 1].min())/n
#for x in frange (dataset[:, 0].min(), dataset[:, 0].max(), xstep):
#  for y in frange (dataset[:, 1].min(), dataset[:, 1].max(), ystep):
#    datasetfiltered = dataset[(dataset[:,0] > x)
#                             &(dataset[:, 0] < x+xstep)
#                             &(dataset[:, 1] > y)
#                             &(dataset[:, 1] < y+ystep)]
#    if datasetfiltered.shape[0] > 0:
#      datasetfiltered = datasetfiltered[datasetfiltered[:, 2]
#                        >(datasetfiltered[:, 2].min()+ zfiltered)]
#      if datasetfiltered.shape[0] > 0:
#        dataset_Z_filtered = np.concatenate((dataset_Z_filtered,
#                                             datasetfiltered))
#print("dataset_Z_filtered shape", dataset_Z_filtered.shape)

#print some specs of dataset pre
print("Examining Point Format PRE: ")
pointformat = inFilePre.point_format
for spec in inFilePre.point_format:
    print(spec.name)

print("Z range =", datasetpre[:, 2].max() - datasetpre[:, 2].min())
print("Z max =", datasetpre[:, 2].max(), "Z min =", datasetpre[:, 2].min())
print("Y range =", datasetpre[:, 1].max() - datasetpre[:, 1].min())
print("Y max =", datasetpre[:, 1].max(), "Y min =", datasetpre[:, 1].min())
print("X range =", datasetpre[:, 0].max() - datasetpre[:, 0].min())
print("X max =", datasetpre[:, 0].max(), "X min =", datasetpre[:, 0].min())

#print some specs of dataset post
print("Examining Point Format POST: ")
pointformat = inFilePost.point_format
for spec in inFilePost.point_format:
    print(spec.name)

print("Z range =", datasetpost[:, 2].max() - datasetpost[:, 2].min())
print("Z max =", datasetpost[:, 2].max(), "Z min =", datasetpost[:, 2].min())
print("Y range =", datasetpost[:, 1].max() - datasetpost[:, 1].min())
print("Y max =", datasetpost[:, 1].max(), "Y min =", datasetpost[:, 1].min())
print("X range =", datasetpost[:, 0].max() - datasetpost[:, 0].min())
print("X max =", datasetpost[:, 0].max(), "X min =", datasetpost[:, 0].min())

#dataset = preprocessing.normalize(dataset)

#clustering pre using DBSCAN
clusteringpre = DBSCAN(eps=0.70, min_samples=450, leaf_size=1).fit(datasetpre)
core_samples_maskpre = np.zeros_like(clusteringpre.labels_, dtype=bool)
core_samples_maskpre[clusteringpre.core_sample_indices_] = True
labelspre = clusteringpre.labels_
n_clusters_pre = len(set(labelspre)) - (1 if -1 in labelspre else 0)
n_noise_pre = list(labelspre).count(-1)
print("Estimated number of clusters PRE: %d" % n_clusters_pre)
print("Estimated number of noise points PRE: %d" % n_noise_pre)

#clustering post using DBSCAN
clusteringpost = DBSCAN(eps=0.70, min_samples=450, leaf_size=1).fit(datasetpost)
core_samples_maskpost = np.zeros_like(clusteringpost.labels_, dtype=bool)
core_samples_maskpost[clusteringpost.core_sample_indices_] = True
labelspost = clusteringpost.labels_
n_clusters_post = len(set(labelspost)) - (1 if -1 in labelspost else 0)
n_noise_post = list(labelspost).count(-1)
print("Estimated number of clusters POST: %d" % n_clusters_post)
print("Estimated number of noise points POST: %d" % n_noise_post)

j=0
cluster_erratipre=0
# Black removed and is used for noise instead.
figpre = plt.figure()
axpre = figpre.add_subplot(111, projection="3d")
unique_labelspre = set(labelspre)
colors = [plt.cm.Spectral(each)
  for each in np.linspace(0, 1, len(unique_labelspre))]
for k, col in zip(unique_labelspre, colors):
  if k == -1:
    # Black used for noise.
    col = [0, 0, 0, 1]
  class_member_mask = (labelspre == k)
  xyz = datasetpre[class_member_mask & core_samples_maskpre]
  axpre.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=col, marker=".")
  #save cluster in a file
  if j<n_clusters_pre:
    arrayNP = np.vstack(xyz).transpose()
    #minimum cluster size control
    if arrayNP[0].size>1000:
      print("Scrittura file di outputPRE"+str(j-cluster_erratipre)+"...")
      outFile = laspy.file.File("C:/Users/crist/OneDrive/Documenti/Univpm/4° Anno/Computer Vision e Deep Learning/ProgettoUlivi/UliviClusterizzati/outputPRE"+str(j-cluster_erratipre)+".las", mode = "w", header = point_cloud1.header)
      outFile.x = arrayNP[0]
      outFile.y = arrayNP[1]
      outFile.z = arrayNP[2]
      outFile.close()
    else:
      cluster_erratipre=cluster_erratipre+1
    j=j+1
#print number of clusters
plt.title("Estimated number of REAL cluster PRE: "+str(n_clusters_pre-cluster_erratipre) )


i=0
cluster_erratipost=0
# Black removed and is used for noise instead.
figpost = plt.figure()
axpost = figpost.add_subplot(111, projection="3d")
unique_labelspost = set(labelspost)
colors = [plt.cm.Spectral(each)
  for each in np.linspace(0, 1, len(unique_labelspost))]
for k, col in zip(unique_labelspost, colors):
  if k == -1:
    # Black used for noise.
    col = [0, 0, 0, 1]
  class_member_mask = (labelspost == k)
  xyz = datasetpost[class_member_mask & core_samples_maskpost]
  axpost.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=col, marker=".")
  #save cluster in a file
  if i<n_clusters_post:
    arrayNP = np.vstack(xyz).transpose()
    #minimum cluster size control
    if arrayNP[0].size>1000:
      print("Scrittura file di outputPOST"+str(i-cluster_erratipost)+"...")
      outFile = laspy.file.File("C:/Users/crist/OneDrive/Documenti/Univpm/4° Anno/Computer Vision e Deep Learning/ProgettoUlivi/UliviClusterizzati/outputPOST"+str(i-cluster_erratipost)+".las", mode = "w", header = point_cloud1.header)
      outFile.x = arrayNP[0]
      outFile.y = arrayNP[1]
      outFile.z = arrayNP[2]
      outFile.close()
    else:
      cluster_erratipost=cluster_erratipost+1
    i=i+1
#print number of clusters
plt.title("Estimated number of REAL cluster POST: "+str(n_clusters_post-cluster_erratipost) )

#plot all clusters
plt.show()
