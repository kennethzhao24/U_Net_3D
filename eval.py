import mrcfile as mrc
import numpy as np
import warnings
from pathlib import Path
from pycm import ConfusionMatrix
from pycm.pycm_output import table_print, stat_print
from pycm.pycm_param import SUMMARY_CLASS, SUMMARY_OVERALL
from scipy.spatial import distance
from skimage.morphology import dilation
import argparse
import matplotlib.pyplot as plt
import scikitplot as skplt

import warnings
warnings.simplefilter('ignore')

import seaborn as sns
sns.set()

if __name__ == '__main__':

    # Script parameters
    parser = argparse.ArgumentParser(description='SHREC 2020 Evaluation')

    parser.add_argument( '--submission', default='./particle_locations/submission.txt')
    args = parser.parse_args()

    submission = args.submission

    # Conversion dicts
    classes = ['0', '3cf3', '1s3x', '1u6g', '4cr2', '1qvr', '3h84', '2cg9', '3qm1', '3gl1', '3d2f', '4d8q', '1bxn']
    num2pdb = {k: v for k, v in enumerate(classes)}
    pdb2num = {v: k for k, v in num2pdb.items()}

    # Loading ground truth particles
    gt = np.genfromtxt("./data/shrec_2020/shrec2020_full/model_9/particle_locations.txt",dtype='str')
    gt_particles = [('0', 0, 0, 0)]  # start with a "background" particle
    for i in range(len(gt)):
        pdb_id, x, y, z, *_ = gt[i]
        gt_particles.append((pdb_id, int(x), int(y), int(z)))
    n_gt_particles = len(gt_particles) - 1

    # Loading occupancy map (voxel -> particle) and morphologically dilate it as a way to close holes
    with mrc.open("./data/shrec_2020/shrec2020_full/model_9/occupancy_mask.mrc", permissive=True) as f:
        occupancy_map = dilation(f.data)

    pred = np.genfromtxt('./particle_locations/submission.txt', dtype='str')
    predicted_particles = []
    for i in range(len(pred)):
        pdb, x, y, z, *_ = pred[i]
        predicted_particles.append((pdb, int(round(float(x))), int(round(float(y))), int(round(float(z)))))
    n_predicted_particles = len(predicted_particles)

    # Init of some vars for statistics
    # number of particles that were predicted to be outside of tomogram
    n_clipped_predicted_particles = 0 
    # reported classes and distances for each GT particle
    found_particles = [[] for _ in range(len(gt_particles))] 

    for p_i, (p_pdb, *coordinates) in enumerate(predicted_particles):

        p_x, p_y, p_z = np.clip(coordinates, (0, 0, 0), (511, 511, 199))

        if [p_x, p_y, p_z] != coordinates:
            n_clipped_predicted_particles += 1
        
        # Find ground truth particle at the predicted location
        p_gt_id = int(occupancy_map[p_z, p_y, p_x])
        p_gt_pdb, p_gt_x, p_gt_y, p_gt_z = gt_particles[p_gt_id]

        # Compute distance from predicted center to real center
        p_distance = np.abs(distance.euclidean((p_x, p_y, p_z), (p_gt_x, p_gt_y, p_gt_z)))

        # Register found particle, a class it is predicted to be and distance from predicted center to real center
        found_particles[p_gt_id].append((p_pdb, p_distance)) 

    # Compute localization statistics
    n_prediction_missed = len(found_particles[0])
    n_prediction_hit = sum([len(p) for p in found_particles[1:]])
    n_unique_particles_found = sum([int(p >= 1) for p in [len(p) for p in found_particles[1:]]])
    n_unique_particles_not_found = sum([int(p == 0) for p in [len(p) for p in found_particles[1:]]])
    n_unique_particle_with_multiple_hits = sum([int(p > 1) for p in [len(p) for p in found_particles[1:]]])

    localization_recall = n_unique_particles_found / n_gt_particles
    localization_precision = n_unique_particles_found / n_predicted_particles
    localization_f1 = 1 / ((1/localization_recall + 1/localization_precision) / 2)
    localization_miss_rate = n_unique_particles_not_found / n_gt_particles
    localization_avg_distance = sum([p[0][1] for p in found_particles[1:] if len(p) > 0]) / n_unique_particles_found

    # Compute classification statistics and confusion matrix
    gt_particle_classes = np.asarray([pdb2num[p[0]] for p in gt_particles[1:]], dtype=int)
    predicted_particle_classes = np.asarray([pdb2num[p[0][0]] if p else 0 for p in found_particles[1:]], dtype=int)
    confusion_matrix = ConfusionMatrix(actual_vector=gt_particle_classes, predict_vector=predicted_particle_classes)
    confusion_matrix.relabel(num2pdb)

    lut_classes = np.asarray(classes)
    skplt.metrics.plot_confusion_matrix(lut_classes[gt_particle_classes],
                                                lut_classes[predicted_particle_classes],
                                                labels=['1s3x', '3qm1', '3gl1', '3h84', '2cg9', '3d2f', '1u6g', '3cf3', '1bxn', '1qvr', '4cr2', '4d8q'],
                                                figsize=(14, 14), text_fontsize=18, hide_zeros=True,
                                                hide_counts=True)
    plt.savefig('plain_cm.png')

    skplt.metrics.plot_confusion_matrix(lut_classes[gt_particle_classes],
                                                lut_classes[predicted_particle_classes],
                                                labels=['1s3x', '3qm1', '3gl1', '3h84', '2cg9', '3d2f', '1u6g', '3cf3', '1bxn', '1qvr', '4cr2', '4d8q'],
                                                figsize=(14, 14), text_fontsize=18, hide_zeros=True,
                                                hide_counts=False)
    plt.savefig('numbers_cm.png')

    # Prepare confusion matrix prints
    confusion_matrix_table = table_print(confusion_matrix.classes, confusion_matrix.table)
    confusion_matrix_stats = stat_print(confusion_matrix.classes, confusion_matrix.class_stat,
                                            confusion_matrix.overall_stat, confusion_matrix.digit,
                                            SUMMARY_OVERALL, SUMMARY_CLASS)

    # Format confusion matrix and stats
    confusion_matrix_table = '\t'.join(confusion_matrix_table.splitlines(True))
    confusion_matrix_stats = '\t'.join(confusion_matrix_stats.splitlines(True))
    

    # Construct a report and write it
    report = f'\n\t### Localization\n' \
                 f'\tSubmission has {n_predicted_particles} predicted particles\n' \
                 f'\tTomogram has {n_gt_particles} particles\n' \
                 f'\tTP: {n_unique_particles_found} unique particles found\n' \
                 f'\tFP: {n_prediction_missed} predicted particles are false positive\n' \
                 f'\tFN: {n_unique_particles_not_found} unique particles not found\n' \
                 f'\tThere was {n_unique_particle_with_multiple_hits} particles that had more than one prediction\n' \
                 f'\tThere was {n_clipped_predicted_particles} predicted particles that were outside of tomo bounds\n' \
                 f'\tAverage euclidean distance from predicted center to ground truth center: {localization_avg_distance}\n' \
                 f'\tRecall: {localization_recall:.5f}\n' \
                 f'\tPrecision: {localization_precision:.5f}\n' \
                 f'\tMiss rate: {localization_miss_rate:.5f}\n' \
                 f'\tF1 score: {localization_f1:.5f}\n' \
                 f'\n\t### Classification\n' \
                 f'\t{confusion_matrix_table}\n' \
                 f'\t{confusion_matrix_stats}\n\n\n'
    print(report)
    confusion_matrix.save_html('confusion_matrix')
