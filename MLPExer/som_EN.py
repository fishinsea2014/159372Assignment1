import numpy as np
import matplotlib.pyplot as plt
#
def w_init(row, col, data_rows):
    w1 = np.zeros((row, col, data_rows))
    for i in range(row):
        for j in range(col):
            w1[i][j] = np.random.rand(1, data_rows)
    return w1
def data_load():
    f = open('iris.txt')
    data = []
    data1 = []
    for line in f.readlines():
        data.append(line.strip().split(','))
    print (data)
    data.pop(0)
    rows = len(data[0])-1
    cols = len(data)
    for col in range(cols):
        data1.append(data[col][0:rows])
    data1 = np.array(data1, dtype=np.float32)
    for col in range(cols):
        data2 = data1[col]/np.sqrt(sum(data1[col] ** 2))
        data1[col] = data2
    print (data1)
    return data1, cols, rows
def update_w(data, w, alpha_init, zeta_init, max_iter):
    for iter1 in range(max_iter):
        print('Calculating..................')
        w_m, w_n, *p = np.shape(w)
        d = np.zeros((w_m, w_n), dtype=np.float32)
        t = np.random.randint(0, 150)
        alpha = alpha_init * np.exp(- iter1/1000)
        zeta = zeta_init * np.exp(- iter1/500)
        for i in range(w_m):
            for j in range(w_n):
                d1 = np.sum((w[i][j] - data[t]) ** 2)
                d[i][j] = np.sum(d1)
        min_d = 2
        for i in range(w_n):
            for j in range(w_m):
                if min_d >= d[i][j]:
                    min_d = d[i][j]
                else:
                    min_d = min_d
        ind1, ind2, *p = np.where(d == min_d)
        for i in range(w_n):
            for j in range(w_m):
                di = (i - ind1) ** 2 + (j - ind2) ** 2
                h = np.exp(- (di / (2 * zeta ** 2)))
                w[i][j] = w[i][j] + alpha * h * (data[t] - w[i][j])
    print('Done')
    return w
def draw(data, w):
    t1 = np.linspace(-0.5, 9.5, 11)
    xx, yy = np.meshgrid(t1, t1)
    w_m, w_n, *p = np.shape(w)
    for t in range(cols):
        d = np.zeros((w_n, w_m), dtype=np.float32)
        for i in range(w_m):
            for j in range(w_n):
                d1 = np.sum((w[i][j] - data[t]) ** 2)
                d[i][j] = np.sum(d1)
        min_d = 2
        for i in range(w_n):
            for j in range(w_m):
                if min_d >= d[i][j]:
                    min_d = d[i][j]
                else:
                    min_d = min_d
        ind1, ind2, *p = np.where(d == min_d)
        plt.plot(t1, xx, 'w')
        plt.plot(xx, t1, 'w')
        plt.title('Flower Cluster')
        if t < 50:
            plt.plot(ind1[0], ind2[0], 'r*')
        if (50 < t) & (t < 100):
            plt.plot(ind1[0], ind2[0], 'b*')
        if (100 < t) & (t < 150):
            plt.plot(ind1[0], ind2[0], 'g*')
    plt.show()
if __name__ == '__main__':
    data, cols, rows = data_load()
    w = w_init(10, 10, rows)
    w = update_w(data, w, 0.2, 3, 5000)
    draw(data, w)





