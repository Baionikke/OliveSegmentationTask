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
