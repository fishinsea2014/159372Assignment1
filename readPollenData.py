__author__ = 'QSG'
def readDatFild():
    # Extract data from .dat files.
    import os
    filelist=os.listdir(".\\Pollens")
    data=[]
    type=0
    for files in filelist:
        f=open('.\\Pollens\\'+files,'r')
        for lines in f.readlines():
            l=[]
            l.extend(lines.strip().split(" "))
            # print l
            ln=[]
            for n in l:
                if n!="":
                    ln.append(float(n.strip()))
            ln.append(type)
            # print d
            data.append(ln)
        type+=1
        f.close()
    return data

    # The data set has 650 lines, 13 types of pollen, each pollen has 50 lines data, and each item has 43 numbers.