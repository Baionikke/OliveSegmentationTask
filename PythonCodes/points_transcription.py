import numpy as np
import laspy as lp
import matplotlib.pyplot as plt
import glob
import os
import pylas


#Open file to shasum
shasum = []
shasum2 = []

with open('./Shasum.txt', 'r') as p:
    for line in p:
        shasum.append(line[0:-2] + '.pts')

with open('./Shasum2.txt', 'r') as p:
    for line in p:
        shasum2.append(line[0:-2] + '.pts')

# Points transcription
i=0
for folder in glob.glob(os.path.join('/home/baionikke/Downloads/data-deep/pre-potatura/', 'NO*')):
    for file in glob.glob(os.path.join(folder, '*label.las')):

        # Upload file
        input_path = file
        point_cloud = lp.file.File(file, mode="r")
        print("File caricato.")

        # Point extraction
        points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()

        with open('/home/baionikke/Downloads/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0/03888888/points/' + shasum[i], 'w') as f:
            for x in points:
                f.write(str(x) + '\n')
    i += 1

i=0
for folder in glob.glob(os.path.join('/home/baionikke/Downloads/data-deep/post-potatura/', 'SI*')):
    for file in glob.glob(os.path.join(folder, '*label.las')):

        # Upload file
        input_path = file
        point_cloud = lp.file.File(file, mode="r")
        print("File caricato.")

        # Point extraction
        points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
        
        with open('/home/baionikke/Downloads/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0/03911111/points/' + shasum2[i], 'w') as f:
            for x in points:
                f.write(str(x) + '\n')
    i += 1




'''
#CROWN#
input_path = '/home/baionikke/Downloads/data-deep/pre-potatura/NO3/NO3-chioma.las'
point_cloud = lp.file.File(input_path, mode="r")
print("File caricato.")
points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
with open('chioma.txt', 'w') as f:
    for x in points:
        f.write(str(x) + '\n')

#LABEL#
input_path = '/home/baionikke/Downloads/data-deep/pre-potatura/NO3/NO3-label.las'
point_cloud = lp.file.File(input_path, mode="r")
print("File caricato.")
points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
with open('label.txt', 'w') as f:
    for x in points:
        f.write(str(x) + '\n')

#TRUNK#
input_path = '/home/baionikke/Downloads/data-deep/pre-potatura/NO3/NO3-tronco.las'
point_cloud = lp.file.File(input_path, mode="r")
print("File caricato.")
points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
with open('tronco.txt', 'w') as f:
    for x in points:
        f.write(str(x) + '\n')


for folder in glob.glob(os.path.join('/home/baionikke/Downloads/data-deep/pre-potatura/', 'NO*')):
    for file in glob.glob(os.path.join(folder, '*label.las')):
        print(file)

        # Caricamento file
        input_path = file
        point_cloud = pylas.read(file)
        print("File caricato.")

        points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()

        a = point_cloud.Label
        b = point_cloud.x
        for x in a,b:
            print(x)
'''

#Label transcription
i=0
for folder in glob.glob(os.path.join('/home/baionikke/Downloads/data-deep/pre-potatura/', 'NO*')):
    for file in glob.glob(os.path.join(folder, '*label.las')):

        # Upload file
        input_path = file
        point_cloud = pylas.read(file)
        print("File caricato.")

        #points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()

        with open('/home/baionikke/Downloads/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0/03888888/points_label/' + shasum[i], 'w') as f:
            for x in point_cloud.Label:
                f.write(str(x) + '\n')
    i += 1

i=0
for folder in glob.glob(os.path.join('/home/baionikke/Downloads/data-deep/post-potatura/', 'SI*')):
    for file in glob.glob(os.path.join(folder, '*label.las')):

        # Upload file
        input_path = file
        point_cloud = pylas.read(file)
        print("File caricato.")

        #points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()

        with open('/home/baionikke/Downloads/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0/03911111/points_label/' + shasum2[i], 'w') as f:
            for x in point_cloud.Label:
                f.write(str(x) + '\n')
    i += 1
