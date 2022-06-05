from isaac.utils import  get_obj_file
import os
root_path = '/home/sujc/code/vcpd-master/data/train/gym_test_2'
success = 0
grasps = 0
mode = 'train'
category ='ABCDEFG' if mode == 'eval' else 'ABCDEFGHIJKLMNOPQRSTUVWX'
suc_index = {}
obj_index = {}
for i in range(len(category)):
    suc_index[category[i]] = 0
    obj_index[category[i]] = 0
for file in os.listdir(root_path):
    path = os.path.join(root_path, file)
    result = get_obj_file(path)
    success += int(result[1])
    suc_index[file[0]] += int(result[1])
    grasps += int(result[0])
    obj_index[file[0]] += int(result[0])

print('{} grasps, {} success, {:.2f}%'.format(grasps, success,
                                            success / grasps * 100))
for i in range(len(category)):
    key = category[i]
    if obj_index[key]!=0:
        print("{}: {} grasps, {} success, {:.2f}%".format(key,
                                                       obj_index[key],
                                                       suc_index[key],
                                                       100*suc_index[key]/obj_index[key]))
