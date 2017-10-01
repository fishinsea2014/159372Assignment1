__author__ = 'QSG'

import random

test=[]
for i in range(9):
    item=[random.randint(-9,9)+random.randint(-9,9)/10.0 for _ in range(5)]
    test.append(item)

print test