__author__ = 'QSG'
import os
filelist=os.listdir(".\\Pollens")
data=[]
type=0
for files in filelist:
    f=open('.\\Pollens\\'+files,'r')
    for lines in f.readlines():
        l=[]
        l.extend(lines.strip().split(" "))
        print l
        ln=[]
        for n in l:
            if n!="":
                ln.append(float(n.strip()))
        ln.append(type)
        # print d
        data.append(ln)
    type+=1
    f.close()
# The data set has 650 items, 13 types of pollen, each pollen has 50 items data, and each item has 43 numbers.
lenl=len(data[0])
print 'length of line is :', lenl

totaldata=0
for item in data:
    if len(item)!=lenl:
        print item
    totaldata+=1
print totaldata


# file1=open(".\\Pollens\\pollen1.dat",'r')
# strdata=[]
# for lines in file1:
#     strdata.append(lines.split('\t'))
#
# for items in strdata:
#     print (items)
#     i=[]
#     i.append(items.split(' '))
#     strdata.append(i)
# print strdata
