from torch.utils.data import Dataset
from functools import reduce
import numpy as np
import torch
import os


class CPNDataset(Dataset):
    def __init__(self, root_dir, cfg, padding=True, shuffle=False, transform=None):
        self.root_dir = root_dir
        self.padding = padding
        self.shuffle = shuffle
        self.transform = transform
        self.resolution = cfg['sdf']['resolution']
        self.origin = np.array([[cfg['sdf']['x_min'], cfg['sdf']['y_min'], cfg['sdf']['z_min']]], dtype=np.float32)
        self.num_sample = cfg['train']['num_sample']
        self.scene_list = reduce(lambda x, y: x+y,
                                 [list(map(lambda folder: [folder, idx],
                                           os.listdir(root_dir))) for idx in cfg['sdf']['save_volume']])
        if shuffle:
            np.random.shuffle(self.scene_list)

    def __getitem__(self, idx):
        scene_id = self.scene_list[idx]
        prefix = os.path.join(self.root_dir, scene_id[0], '{:04d}_'.format(scene_id[1]))
        sample = dict()
        sample['sdf_volume'] = np.load(prefix + 'sdf_volume.npy')  # * self.resolution * 5
        sample['occupancy_volume'] = np.zeros_like(sample['sdf_volume'])
        sample['occupancy_volume'][sample['occupancy_volume'] <= 0] = 1
        ids_pcp1, ids_pcp2 = np.load(prefix+'pos_contact1.npy'), np.load(prefix+'pos_contact2.npy')
        ids_ncp1, ids_ncp2 = np.load(prefix+'neg_contact1.npy'), np.load(prefix+'neg_contact2.npy')
        num_pos, num_neg = ids_pcp1.shape[0], ids_ncp1.shape[0]
        sample['ids_cp1'] = np.concatenate([ids_pcp1, ids_ncp1], axis=0)
        sample['ids_cp2'] = np.concatenate([ids_pcp2, ids_ncp2], axis=0)
        sample['label'] = np.concatenate([np.ones(num_pos),
                                          np.zeros(num_neg)]).reshape(-1, 1)  # N * 1
        sample['mask'] = np.ones_like(sample['label'])
        sample['origin'] = self.origin.reshape(-1)  # (3,)
        sample['resolution'] = np.array([self.resolution])  # (1,)
        if self.padding:
            sample = self.sample_pts(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.scene_list)

    def sample_pts(self, sample):
        num_pts = sample['ids_cp1'].shape[0]
        if num_pts >= self.num_sample:
            num_pos = np.sum(sample['label']).astype(int)
            num_neg = num_pts - num_pos
            half1 = self.num_sample // 2
            half2 = self.num_sample - half1
            pos_ids, neg_ids = np.arange(num_pos), np.arange(num_pos, num_pts)
            if num_pos <= half1:
                pos_ids, neg_ids = pos_ids, np.random.choice(neg_ids, self.num_sample-num_pos, replace=False)
            elif num_neg <= half2:
                pos_ids, neg_ids = np.random.choice(pos_ids, self.num_sample-num_neg, replace=False), neg_ids
            else:
                pos_ids = np.random.choice(pos_ids, half1, replace=False)
                neg_ids = np.random.choice(neg_ids, half2, replace=False)
            ids = np.concatenate([pos_ids, neg_ids], axis=0)
            sample['mask'] = np.ones(self.num_sample).reshape(-1, 1)  # N' * 1
        else:
            ids = np.arange(num_pts)
            complementary = np.random.choice(ids, self.num_sample-num_pts, replace=True)
            ids = np.concatenate([ids, complementary], axis=0)
            sample['mask'] = np.concatenate([np.ones(num_pts),
                                             np.zeros(self.num_sample-num_pts)]).reshape(-1, 1)
        sample['ids_cp1'] = sample['ids_cp1'][ids]
        sample['ids_cp2'] = sample['ids_cp2'][ids]
        sample['label'] = sample['label'][ids]
        return sample


class Permute(object):
    def __call__(self, sample):
        for k, v in sample.items():
            if k != 'origin' and k != 'resolution':
                sample[k] = np.expand_dims(v, axis=0)
                if 'volume' not in k:
                    sample[k] = np.transpose(sample[k], (2, 1, 0))
        return sample


class Float32(object):
    def __call__(self, sample):
        for k, v in sample.items():
            sample[k] = v.astype(np.float32)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        for k, v in sample.items():
            sample[k] = torch.from_numpy(v)
        return sample
