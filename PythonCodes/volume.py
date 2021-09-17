import numpy as np
import laspy as lp
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt

plt.rcParams['figure.max_open_warning'] = 1000

# Array contenente tutti i volumi + differenza volumi (vhpre|vhbspre|svpre|vhpost|vhbspost|svpost|vhdiff|vhbsdiff|svdiff)
w, h = 9, 10;
arrayvolumi= [[0 for x in range(w)] for y in range(h)]

for contatore in range(20):
    # Caricamento file
    input_path = "C:/Users/crist/OneDrive/Documenti/Univpm/4° Anno/Computer Vision e Deep Learning/ProgettoUlivi/UliviClusterizzati/"
    if(contatore<10):
        dataname = "outputPRE"+str(contatore)
    else:
        dataname = "outputPOST"+str(contatore-10)
    point_cloud = lp.file.File(input_path+dataname+".las", mode="r")
    print(dataname)

    # Estrazione punti
    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()

    # Visualize .las file
    factor = 5
    decimated_points_random = points[::factor]
    ax = plt.axes(projection='3d')
    ax.scatter(decimated_points_random[:,0], decimated_points_random[:,1], decimated_points_random[:,2]) #c = decimated_colors/65535, s=0.01)
    #plt.show()

    ### CONVEX HULL ###
    hull = ConvexHull(points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(points.T[0], points.T[1], points.T[2], "ko")

    for s in hull.simplices:
        s = np.append(s, s[0])  
        ax.plot(points[s, 0], points[s, 1], points[s, 2], "r-")

    for i in ["x", "y", "z"]:
        eval("ax.set_{:s}label('{:s}')".format(i, i))

    print("CONVEX HULL VOLUME: " + str(hull.volume))
    #plt.show()

    
    ### CONVEX HULL BY SLICE ###
    point_cloud_ordered = np.sort(points, axis=0)
    h_min = np.min(point_cloud.z)
    h_max = np.max(point_cloud.z)
    j = 0
    fig2 = plt.figure()

    while(h_min < h_max):
        point_cloud_subpoints = []
        conta=0

        for i in range(len(point_cloud_ordered)):
            if(point_cloud_ordered[i,2] > h_min and point_cloud_ordered[i,2] < h_min+0.2):
                prova = point_cloud_ordered[i]
                point_cloud_subpoints.append(prova)
                conta+=1

        if(len(point_cloud_subpoints) < 4): break
        hull2 = ConvexHull(point_cloud_subpoints)
        #print(hull2.volume*100000)
        j += hull2.volume*100000
        h_min += 0.2
        #print(conta)

    #print("CONVEX HULL BY SLICE VOLUME: " + str(j))

    ### SECTION VOLUME ###
    def tetrahedron_volume(a, b, c, d):
        return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6

    tri = Delaunay(points)
    tets = tri.points[tri.simplices]
    vol = np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1], 
                                    tets[:, 2], tets[:, 3]))
    print("SECTION VOLUME: " + str(vol))

    ### SCRITTURA ARRAY X SCRITTURA VOLUMI SU FILE (vhpre|vhbspre|svpre|vhpost|vhbspost|svpost|vhdiff|vhbsdiff|svdiff)
    if(contatore<10):
        arrayvolumi[contatore][0] = hull.volume
        arrayvolumi[contatore][1] = j
        arrayvolumi[contatore][2] = vol
    else:
        arrayvolumi[contatore-10][3] = hull.volume
        arrayvolumi[contatore-10][4] = j
        arrayvolumi[contatore-10][5] = vol
for i in range(10):
    arrayvolumi[i][6] = arrayvolumi[i][0] - arrayvolumi[i][3]
    arrayvolumi[i][7] = arrayvolumi[i][1] - arrayvolumi[i][4]
    arrayvolumi[i][8] = arrayvolumi[i][2] - arrayvolumi[i][5]

file_output_vol = open("C:/Users/crist/OneDrive/Documenti/Univpm/4° Anno/Computer Vision e Deep Learning/ProgettoUlivi/UliviClusterizzati/VolumiCompleti.txt", "w")
for i in range(10):
    file_output_vol.write("\n\nULIVO " + str(i))
    file_output_vol.write("\nCONVEX HUL " + str(i) + " : (PRE|POST|DIFF) " + str(arrayvolumi[i][0]) + " | " + str(arrayvolumi[i][3]) + " | " + str(arrayvolumi[i][6]))
    #file_output_vol.write("\nCONVEX HUL BY SLICE " + str(i) + " : (PRE|POST|DIFF) " + str(arrayvolumi[i][1]) + " | " + str(arrayvolumi[i][4]) + " | " + str(arrayvolumi[i][7]))
    file_output_vol.write("\nSECTION VOLUME " + str(i) + " : (PRE|POST|DIFF) " + str(arrayvolumi[i][2]) + " | " + str(arrayvolumi[i][5]) + " | " + str(arrayvolumi[i][8]))
file_output_vol.close
"""
### VOXEL VOLUME ###
factor = 150
decimated_points = points[::factor]

voxel_size = 3
nb_vox = np.ceil((np.max(points, axis=0) - np.min(points, axis=0))/voxel_size)
non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((points - np.min(points, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
idx_pts_vox_sorted = np.argsort(inverse)
voxel_grid = {}
grid_barycenter,grid_candidate_center = [],[]
last_seen = 0

for idx,vox,c in enumerate(non_empty_voxel_keys):
    voxel_grid[tuple(vox)] = points[idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]]
    grid_barycenter.append(np.mean(voxel_grid[tuple(vox)], axis=0))
    grid_candidate_center.append(voxel_grid[tuple(vox)][np.linalg.norm(voxel_grid[tuple(vox)]-np.mean(voxel_grid[tuple(vox)], axis=0), axis=1).argmin()])
    last_seen+=nb_pts_per_voxel[idx]

#ax = plt.axes(projection='3d')
#ax.scatter(decimated_points[:,0], decimated_points[:,1], decimated_points[:,2])#, c = decimated_colors/65535, s=0.01)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------plt.show()

print("VOXEL VOLUME: ")
"""
