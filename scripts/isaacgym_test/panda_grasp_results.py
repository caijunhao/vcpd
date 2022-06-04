import os
import numpy as np

path = '../log/cpn'
name = ['primitive', 'random', 'kit']

for i in range(3):
    print('%s:' % name[i])
    for j in range(5, 25, 5):
        succ = 0
        count = 0
        print('%s objs:' % j)
        for k in range(1, 11):
            result = np.loadtxt(path+'/%s_%s_%s.txt' % (name[i], j, k))
            succ += result[0]
            count += result[1]
        print('success: {}, count: {}, {:0.2f}%'.format(succ, count, succ/count*100))
