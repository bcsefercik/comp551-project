import os
import pickle

train_split = 'splits/scannetv2_train.txt'
val_split   = 'splits/scannetv2_val.txt'
test_split  = 'splits/scannetv2_test.txt'
scannet_dir = 'ScanNetV2/scans/'
output_dir  = 'dataset/scannetv2/'
train_ext = ['_vh_clean_2.ply','_vh_clean_2.labels.ply','_vh_clean_2.0.010000.segs.json', '.aggregation.json']
test_ext  = ['_vh_clean_2.ply']

train_scenes = open(train_split, 'r')
Lines        = train_scenes.readlines()

all_mv_commands = []

i = 1
for line in Lines: 
    print("Training Preparing... ", i/len(Lines),'%')
    line = line.rstrip('\n')
    for ext in train_ext:
        command =  'mv ' + scannet_dir + line + '/' + line + ext + ' ' + output_dir + 'train/'
        print(command)
        all_mv_commands.append(command)
        os.system(command)

    i = i+1

val_scenes = open(val_split, 'r')
Lines        = val_scenes.readlines()


i = 1
for line in Lines: 
    print("Validation Preparing... ", i/len(Lines),'%')
    line = line.rstrip('\n')
    for ext in train_ext:
        command =  'mv ' + scannet_dir + line + '/' + line + ext + ' ' + output_dir + 'val/'
        print(command)
        all_mv_commands.append(command)
        os.system(command)

    i = i+1

test_scenes = open(test_split, 'r')
Lines        = test_scenes.readlines()


i = 1
for line in Lines: 
    print("Testing Preparing... ", i/len(Lines),'%')
    line = line.rstrip('\n')
    for ext in test_ext:
        command =  'mv ' + scannet_dir + line + '/' + line + ext + ' ' + output_dir + 'test/'
        print(command)
        all_mv_commands.append(command)
        os.system(command)
    i = i+1

with open("all_mv_commands.txt", "wb") as fp: 
    pickle.dump(all_mv_commands, fp)