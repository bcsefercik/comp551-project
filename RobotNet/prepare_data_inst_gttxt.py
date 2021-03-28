'''
Generate instance groundtruth .txt files (for evaluation)
'''

import numpy as np
import glob
import torch
import pickle
import os

import ipdb

semantic_label_idxs = [0,1]
semantic_label_names = ['background', 'arm']


def get_data(file_names,index):
    curr_file_name = file_names[index]

    with open(curr_file_name, 'rb') as f:
        x = pickle.load(f)
        return x



if __name__ == '__main__':
    split = 'test'
    file_names = sorted(glob.glob('dataset/alivev1/{}/*.pickle'.format(split)))
    #rooms = [torch.load(i) for i in files]

    if not os.path.exists('dataset/alivev1/' + split + '_gt'):
        os.mkdir('dataset/alivev1/' + split + '_gt')

    for i in range(len(file_names)):

        # xyz, rgb, label, instance_label = get_data(file_names, i)
        xyz, rgb, label, instance_label, _ = get_data(file_names, i)
    
        scene_name = file_names[i].split('/')[-1].split('.pickle')[0]
        print('{}/{} {}'.format(i + 1, len(file_names), scene_name))

        ipdb.set_trace()

        instance_label_new = np.zeros(instance_label.shape, dtype=np.int32)  # 0 for unannotated, xx00y: x for semantic_label, y for inst_id (1~instance_num)

        instance_num = int(instance_label.max()) + 1
        for inst_id in range(instance_num):
            instance_mask = np.where(instance_label == inst_id)[0]
            sem_id = [int(i) for i in label[instance_mask] ]
            if(sem_id == -100): sem_id = 0
            #semantic_label = semantic_label_idxs[sem_id]
            #print('semantic label is: ', semantic_label)
            instance_label_new[instance_mask] = np.array(sem_id) * 1000 + inst_id + 1

        np.savetxt(os.path.join('dataset/alivev1/' + split + '_gt', scene_name + '.txt'), instance_label_new, fmt='%d')





