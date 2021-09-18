# OliveSegmentationTask
Segmentation task of olive trees in order to estimate crown volume before and after pruning and calculate the difference between them.

In order to see the PowerPoint presentation, visit and download it from the pointnet dataset (last link below).

Step to follow to estiamte crown volume. Run the following python scripts in this sequence*:
1) oliveDownsamp = DOWNSAMPLING OLIVES
2) oliveCode = DBSCAN (CLUSTERIZATION)
3) volume = CROWN OLIVES VOLUME CALCULATION

*NB:these codes have been running on local; so you have to adjust the path within the reading and writing files.

The segmantation task is to be runned on a different location beacuase it does not participate in the volume calculation due to inaccurate results (dataset too small).

Link to the colab script at the url: https://colab.research.google.com/drive/132XiPrUXfRpviJzv9Z6-qQLraH_IE1r1?usp=sharing

Link to the pointnet dataset at the url: https://drive.google.com/drive/folders/1A7AB_OsJJw3sCRIMsgdqZmYz9b_8-K_O?usp=sharing

Step to follow to run the colab script:
1) Download the pointnet dataset from the link and upload to your local google drive
2) Upload the dataset on /gdrive/MyDrive/ (everything is set up to run automatically with this path)
3) Run the colab script
