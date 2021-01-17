import os
import pickle

folder_name       = '_gitignore/Dataset/p1/full_light/robotNet/test_raw_pose/'
output_name       = '_gitignore/Dataset/p1/full_light/robotNet/test_raw_pose_p2/'
file_names        = [name for name in os.listdir(folder_name) if os.path.isdir(folder_name)]

i = 0
for file_name in file_names:
    print(i,'/' ,len(file_names))
    filehandler = open(folder_name + file_name,'rb')
    p3_object = pickle.load(filehandler)
    filehandler.close()
    filehandler = open(output_name+file_name,"wb")
    pickle.dump(p3_object, filehandler, protocol=2)
    filehandler.close()
    i = i+1