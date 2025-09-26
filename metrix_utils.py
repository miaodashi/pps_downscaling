import os
import math
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def calculate_mean_std(dataloader) -> tuple:
    """calculate the mean and std of the dataset."""
    labels = []
    for _, label in dataloader:
        labels.append(label)

    labels = torch.cat(labels)
    mean = labels.mean().item()
    std = labels.std().item()

    return mean, std


def eval_metric(evaldata, label) -> dict:
    """
    using for calculate the R2, MAE, correlation_coefficient, RMSE
    dist: r2, r, mae, rmse
    """
    metric = {
        "r2": r2_score(label, evaldata),
        "r": np.corrcoef(label, evaldata)[0, 1],
        "mae": mean_absolute_error(label, evaldata),
        "rmse": np.sqrt(mean_squared_error(label, evaldata)),
    }
    return metric


def draw_points(output, label, savepth: str, name: str, maxv=None, minv=None) -> tuple:
    """Draws a scatter plot of the output and label data, and saves it to a file."""
    if maxv is None and minv is None:
        min_val = min(label)
        max_val = max(label)

        min_val = (min_val // 5) * 5
        max_val = math.ceil(max_val / 5) * 5

        if min_val == max_val:
            min_val = min_val - 5 if min_val - 5 > 0 else 0
            max_val = max_val + 5 if max_val + 5 < 100 else 100
        else:
            min_val = 0 if min_val < 0 else min_val
            max_val = 100 if max_val > 100 else max_val
    else:
        min_val = minv
        max_val = maxv

    xy = np.vstack([label, output])  # 将x和y坐标堆叠成一个二维数组，以便计算密度。
    z = gaussian_kde(xy)(
        xy
    )  # 使用生成的KDE对象来评估每个点的密度，并将这些密度值存储在z中

    fig, ax = plt.subplots()  # 创建了一个图和坐标轴对象

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    plt.xlabel("Measured (%)", fontsize=12)
    plt.ylabel("Predicted (%)", fontsize=12)
    plt.plot([min_val, max_val], [min_val, max_val], "k--")

    scatter = ax.scatter(
        label, output, c=z, s=5, edgecolor=None, cmap="plasma"
    )  # 创建了一个散点图，其中点的颜色和大小由z控制

    fig.colorbar(scatter, ax=ax, label="Density").set_ticks(
        []
    )  # 创建了一个颜色条，显示了密度值的范围, 不显示刻度
    # plt.show()

    plt.savefig(os.path.join(savepth, name + ".png"), dpi=300)
    plt.close()
    return min_val, max_val


def parse_prediction_and_label_from_text(filename):
    # 解析txt文件中的数据
    with open(filename, "r") as f:
        lines = f.readlines()

    # 解析数据
    outputs = []
    labels = []
    for line in lines:
        # 跳过空行和包含"Epoch"的行
        if line.strip() == "" or "Epoch" in line:
            continue

        # 分割数据点
        points = line.split(";")
        for point in points:
            # 跳过空数据点
            if point == "":
                continue

            # 分割预测值和真实值
            output, label = point.split(",")
            outputs.append(float(output))
            labels.append(float(label))

    return outputs, labels


if __name__ == "__main__":
    flag = 2

    if flag == 1:
        # 生成测试数据
        output = np.random.uniform(low=0, high=30, size=(50,))
        label = np.random.uniform(low=0, high=30, size=(50,))

        # 指定保存路径和文件名
        savepth = "./test"
        name = "test_plot"

        # 确保保存路径存在
        if not os.path.exists(savepth):
            os.makedirs(savepth)

        # 调用 draw_points 函数
        draw_points(output, label, savepth, name)
    
    if flag == 2:
        label = np.random.rand(8,1,50,50)
        output = np.random.rand(8,1,50,50)
        
        from torch import nn
        print(nn.MSELoss()(torch.tensor(output), torch.tensor(label)))
        # d = eval_metric(output, label)
        # print(d)
