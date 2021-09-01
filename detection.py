import mrcfile
import numpy as np
import pandas as pd
from scipy.spatial import distance
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import dilation

filter_threshold = 0.5

classes = ['0', '3cf3', '1s3x', '1u6g', '4cr2', '1qvr', '3h84', '2cg9', '3qm1', '3gl1', '3d2f', '4d8q', '1bxn']

print("Generating particle locations....")

for i in range(len(classes)):
    mrc_filepath = './result/shrec_mask/shrec_result_class_{}.mrc'.format(classes[i])
    pred = mrcfile.open(mrc_filepath).data.astype(int)
    pred = np.where(pred < filter_threshold, pred, 1)
    pred = np.where(pred >= filter_threshold, pred, 0)
    pred_mask = dilation(pred)
    pred_mask = label(pred_mask, connectivity=1)
    pred_props = regionprops(pred_mask)

    centroids = np.zeros((len(pred_props), 3), dtype=int)
    for j in range(len(pred_props)):
        centroids[j,:] = [pred_props[j].centroid[2], pred_props[j].centroid[1], pred_props[j].centroid[0]]
    
    class_label = i * np.ones((len(centroids), 1), dtype=int)
    centroids = np.append(class_label, centroids, axis=1)
    centroids = centroids.astype(str)
    centroids[:,0] = classes[i]
    np.savetxt('./particle_locations/class_{}_locations.txt'.format(classes[i]), centroids, delimiter=" ", newline="\n", fmt="%s")
    print("{}/{}".format(i+1, len(classes)))

for i in range(1,len(classes)):
    pred_temp = np.genfromtxt('./particle_locations/class_{}_locations.txt'.format(classes[i]), dtype='str')
    if i == 1:
        pred_final = pred_temp
    else:
        pred_final = np.vstack((pred_final, pred_temp))

np.savetxt('./particle_locations/submission.txt', pred_final, delimiter=" ", newline="\n", fmt="%s")

print("Process complete!")





