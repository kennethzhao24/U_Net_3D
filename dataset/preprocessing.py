import torch.utils.data as data
import numpy as np
import mrcfile
import pandas as pd
import torch
import warnings

# data preparation for shrec20
class SHREC_2020_Dataset(data.Dataset):
    def __init__(self, mode='train', block_size=72, label_type='mask', num_class = 12, random_num=1000):
        self.mode = mode
        self.label_type = label_type
        self.num_class = num_class
        if self.mode == 'train':
            self.data_range = [0, 1, 2, 3, 4, 5, 6, 7]
        elif self.mode == 'val':
            self.data_range = [8]
        else:#test
            self.data_range = [9]
        
        self.shift = block_size // 2

        self.radius = 15

        self.data_volume = []
        self.class_mask = []
        self.location = []

        #to avoid mrcfile warnings
        warnings.simplefilter('ignore')

        for i in range(10):
            with mrcfile.open('./data/shrec2020/model_%d/reconstruction_norm.mrc' % i, permissive=True) as gm:
                self.data_volume.append(gm.data)

            with mrcfile.open('./data/shrec2020/model_%d/class_mask.mrc' % i, permissive=True) as cm:
                self.class_mask.append(cm.data)
            
            self.location.append(pd.read_csv('./data/shrec2020/model_%d/particle_locations.txt' % i, header=None, sep=' '))
        
        self.data_index = []
       
        if not self.mode == 'test':
            for i in self.data_range:
                location = self.location[i]
                for j in range(len(location)):
                    line = location.loc[j]
                    shape = self.data_volume[i].shape
                    sample_point = self.__sample(np.array([line[1], line[2], line[3]]), np.array([shape[2], shape[1], shape[0]]))
                    self.data_index.append(np.concatenate(([i], sample_point)))
        else:
            for i in self.data_range:
                shape = self.data_volumeme[i].shape
                for j in range(shape[0] // (2 * self.shift) + 1):
                    for k in range(shape[1] // (2 * self.shift) + 1):
                        for l in range(shape[2] // (2 * self.shift) + 1):
                            if j == shape[0] // (2 * self.shift):
                                x = shape[0] - self.shift
                            else:
                                x = j * 2 * self.shift + self.shift
                            
                            if k == shape[1] // (2 * self.shift):
                                y = shape[1] - self.shift
                            else:
                                y = k * 2 * self.shift + self.shift
                            
                            if l == shape[2] // (2 * self.shift):
                                z = shape[2] - self.shift
                            else:
                                z = l * 2 * self.shift + self.shift
                            
                            self.data_index.append(np.array([i, z, y, x]))
        
        # add random samples in training set
        if mode == 'train' and random_num > 0:
            print('random samples num:', random_num)
            for j in range(random_num):
                i = np.random.randint(8)
                data_shape = self.data_volumeme[i].shape
                x = np.random.randint(self.shift, data_shape[0] - self.shift)
                y = np.random.randint(self.shift, data_shape[1] - self.shift)
                z = np.random.randint(self.shift, data_shape[2] - self.shift)
                self.data_index.append([i, z, y, x])
        
        self.data = []
        for point in self.data_index:
            i, x, y, z = point
            img = self.data_volume[i][z - self.shift : z + self.shift, y - self.shift : y + self.shift, x - self.shift : x + self.shift]
            label = self.class_mask[i][z - self.shift : z + self.shift, y - self.shift : y + self.shift, x - self.shift : x + self.shift]
            self.data.append([img, label])
        
    def __getitem__(self, index):
        img, label = self.data[index]

        # random 3D rotation for data augmentation
        if self.mode == 'train':
            degree = np.random.randint(4, size=3)
            img = self.__rotation3D(img, degree)
            label = self.__rotation3D(label, degree)
        
        img = np.array(img)
        label = np.array(label)

        # Transform single channel label to multichannel
        label = np.reshape(label, (1, self.shift*2, self.shift*2, self.shift*2))
        label = multiclass_label(label, num_classes=self.num_class)

        img = torch.as_tensor(img).unsqueeze(0)
        label = torch.as_tensor(label).unsqueeze(0)
        return img, label
    
    def __len__(self):
        return len(self.data_index)
    
    def __rotation3D(self, data, degree):
        data = np.rot90(data, degree[0], (0,1))
        data = np.rot90(data, degree[1], (1,2))
        data = np.rot90(data, degree[2], (0,2))
        return data
    
    def __sample(self, point, bound):
        new_point = point + np.random.randint(self.radius - self.shift, self.shift - self.radius + 1, size=3)
        new_point[new_point < self.shift] = self.shift
        new_point[new_point + self.shift > bound] = bound[new_point + self.shift > bound] - self.shift
        return new_point

# transform label to multichannel
def multiclass_label(x, num_classes):
    for i in range(num_classes+1):
        label_temp = x
        label_temp = np.where(label_temp == i, 1, 0)
        if i == 0:
            label_new = label_temp
        else:
            label_new = np.concatenate((label_new, label_temp))
    
    return label_new
