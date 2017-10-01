import numpy as np
import matplotlib.pyplot as plt
# W初始化
def w_init(row, col, data_rows):
    w1 = np.zeros((row, col, data_rows))  # 构造W初始数组
    for i in range(row):
        for j in range(col):
            w1[i][j] = np.random.rand(1, data_rows)  # 为W随机赋值
    return w1
# 数据预处理
def data_load():
    f = open(r'iris.txt')  # 导入数据
    data = []
    data1 = []
    for line in f.readlines():
        data.append(line.strip().split(','))  # 切割TXT文件，构造数据列表
    data.pop(0)
    rows = len(data[0])-1  # 数据列表的行数
    cols = len(data)  # 数据列表的列数
    for col in range(cols):
        data1.append(data[col][0:rows])
    data1 = np.array(data1, dtype=np.float32)  # 转换格式：将列表字符格式转换为数组实数格式
    for col in range(cols):
        data2 = data1[col]/np.sqrt(sum(data1[col] ** 2))
        data1[col] = data2
    return data1, cols, rows
# W更新函数
def update_w(data, w, alpha_init, zeta_init, max_iter):
    for iter1 in range(max_iter):
        print('Calculating..................')
        w_m, w_n, *p = np.shape(w)  # 计算W的行数和列数
        d = np.zeros((w_m, w_n), dtype=np.float32)  # 构造距离d初始数组
        t = np.random.randint(0, 150)  # 随机选取一个输入数据
        alpha = alpha_init * np.exp(- iter1/1000)
        zeta = zeta_init * np.exp(- iter1/500)
        for i in range(w_m):
            for j in range(w_n):
                d1 = np.sum((w[i][j] - data[t]) ** 2)  # 计算输入数据与各神经元的距离
                d[i][j] = np.sum(d1)
        min_d = 2
        for i in range(w_n):
            for j in range(w_m):
                if min_d >= d[i][j]:
                    min_d = d[i][j]
                else:
                    min_d = min_d
        ind1, ind2, *p = np.where(d == min_d)  # 获取最优神经元的位置坐标
        for i in range(w_n):
            for j in range(w_m):
                di = (i - ind1) ** 2 + (j - ind2) ** 2
                h = np.exp(- (di / (2 * zeta ** 2)))
                w[i][j] = w[i][j] + alpha * h * (data[t] - w[i][j])  # 更新各个权值
    print('Done')
    return w
# 数据可视化
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
        plt.plot(xx, t1, 'w')  # 绘制网格
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





