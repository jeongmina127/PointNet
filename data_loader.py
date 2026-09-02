import trimesh
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path
import numpy as np
import random, math

PATH = Path("/mnt/c/Users/Jeongmin Cho/Desktop/ModelNet10/ModelNet10_3_splits")
NUM_WORKERS = os.cpu_count()

folders = [dir for dir in sorted(os.listdir(PATH)) if os.path.isdir(PATH/dir)]
classes = {folder : i for i, folder in enumerate(folders)}


class PointSampler:
    """
    CAD -> pointcloud
    """
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, mesh):
        sampled_points, _ = trimesh.sample.sample_surface(mesh = mesh,count= self.output_size)

        return sampled_points

    
class Normalize:
    def __call__(self, pointcloud):
        centroid = np.mean(pointcloud, axis = 0)
        norm_points = pointcloud - centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(pointcloud)**2, axis = -1)))
        norm_points /= furthest_distance

        return norm_points


class RandRotation_z:
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        theta = random.random() * 2 * math.pi
        rot_matrix = np.array([[math.cos(theta), -math.sin(theta), 0],
                              [math.sin(theta), math.cos(theta), 0],
                              [0,               0,               1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T

        return rot_pointcloud


class RandomNoise:
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        noise_rate = 0.2
        # numpy.random.normal(loc=0.0, scale=1.0, size=None)
        noise = np.random.normal(0, noise_rate, pointcloud.shape)
        noise_pointcloud = pointcloud + noise

        return noise_pointcloud


class ToTensor:
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        return torch.from_numpy(pointcloud)


def default_transforms():
    return transforms.Compose([
        PointSampler(1024),
        Normalize(),
        ToTensor()
        ])


class PointcloudData(Dataset):

    def __init__(self, root_dir, valid = False, folder = 'train', transform = default_transforms()):
        self.root_dir = Path(root_dir)
        folders = [dir for dir in sorted(os.listdir(self.root_dir)) if os.path.isdir(self.root_dir/dir)]
        self.classes = {folder : i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []

        for category in self.classes.keys():
            new_dir = root_dir / Path(category) / folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def preprocess(self, pcd_path):
        mesh = trimesh.load(pcd_path)
        pointcloud = self.transforms(mesh)
        return pointcloud

    def __getitem__(self, index):
        pcd_path = self.files[index]['pcd_path']
        category = self.files[index]['category']

        pointcloud = self.preprocess(pcd_path)

        return {'pointcloud':pointcloud,
                'category': self.classes[category]}

    
def train_transforms():
    return transforms.Compose([
        PointSampler(1024),
        Normalize(),
        RandRotation_z(),
        RandomNoise(),
        ToTensor()])


    
# train_ds =  PointcloudData(root_dir=PATH,transform = train_transforms())
# valid_ds = PointcloudData(root_dir=PATH,folder='val', transform= default_transforms(), valid= False)
# test_ds = PointcloudData(root_dir=PATH, valid = False, folder = 'test',transform=default_transforms())

# print('Train dataset size: ', len(train_ds))
# print('Valid dataset size: ', len(valid_ds))
# print('Test dataset size: ', len(test_ds))
# print('Number of classes: ', len(train_ds.classes))
# print(type(train_ds))
# print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())