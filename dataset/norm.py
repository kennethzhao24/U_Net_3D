import mrcfile
import numpy as np
import warnings
import torch

warnings.simplefilter('ignore')

reconstruction_volume = []
for i in range(10):
    with mrcfile.open('./data/shrec2020/model_%d/reconstruction.mrc' % i, permissive=True) as gm:
        reconstruction_volume.append(gm.data)

reconstruction_volume = torch.Tensor(np.array(reconstruction_volume))

min_value = reconstruction_volume.min()

reconstruction_volume = reconstruction_volume - min_value

max_value = reconstruction_volume.max()

reconstruction_volume_norm = reconstruction_volume / max_value

for i in range(10):
    reconstruction_norm = mrcfile.new('./data/shrec2020/model_%d/reconstruction_norm.mrc' % i, overwrite=True)
    reconstruction_norm.set_data(reconstruction_volume_norm[i][156:356].to_numpy())
    reconstruction_norm.close()

