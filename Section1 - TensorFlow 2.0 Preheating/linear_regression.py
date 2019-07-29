import numpy as np
# 求解 y = wx + b，定义 loss 函数
# 参数 b 和 w 分别为偏秩和斜率，points 为点的数组 [(x0,y0),(x1,y1),……,(xn,yn)]
def compute_loss_for_line_given_points(b, w, points):
    totalLoss = 0
    N = float(len(points))
    for i in range(0, len(points)):
        #这是 numpy 一种特殊的数组取值方法，等同于 points[i][0]
        x = points[i, 0]
        y = points[i, 1]
        # 计算 (w * x + b - y) 的平方和
        totalLoss += (w * x + b - y) ** 2
    # 损失函数 loss 取 1/N
    return totalLoss / N

# 定义 gradient descent 算法
# b_current 和 w_current 是上一次调优的 b 和 w，learningRate 是衰减因子 lr
def set_gradient_descent(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # loss 对 b 的偏导数求和
        b_gradient += (2/N) * ((w_current * x + b_current) - y)
        # loss 对 w 的偏导数求和
        w_gradient += (2/N) * x * ((w_current * x + b_current) - y)
    # 更新调优的 b 和 w
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]

# 定义迭代函数，不断地优化 b 和 w
# starting_b 和 starting_w 为预设的偏秩和斜率，num_iterations 是迭代的次数
def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    # 不断地执行 gradient descent 算法
    for i in range(num_iterations):
        b, w = set_gradient_descent(b, w, np.array(points), learning_rate)
    return [b, w]


def run():
    # 这里使用一个随机的数据集，共有100个点，即本节第一张图所示的数据
    points = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0 # 预设的偏秩
    initial_w = 0 # 预设的斜率
    num_iterations = 1000 # 设置循环1000次
    # 打印调优之前的 b、w 和误差
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}"
          .format(initial_b, initial_w,
                  compute_loss_for_line_given_points(initial_b, initial_w, points))
          )
    print("Running...")
    # 得到最优的 b 和 w
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    # 打印调优之后的 b、w 和误差
    print("After gradient descent b = {0}, w = {1}, error = {2}".
          format(b, w,
                 compute_loss_for_line_given_points(b, w, points))
          )

if __name__ == '__main__':
    run()