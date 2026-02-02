import numpy as np
import matplotlib.pyplot as plt

def visualize_grid(path):
    # 读取 npy 文件
    grid = np.load(path)
    print(f"载入地图：{path}，形状：{grid.shape}")

    # 检查是否为二维
    if grid.ndim != 2:
        raise ValueError("读取的 npy 不是二维数组！")

    h, w = grid.shape

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("栅格地图可视化")

    # 显示栅格：0=白，1=黑
    im = ax.imshow(
        grid,
        cmap='gray_r',   # 0 白, 1 黑
        vmin=0,
        vmax=1,
        origin='lower'   # 原点在左下角
    )

    # 画网格线
    ax.set_xticks(np.arange(-0.5, w, 1))
    ax.set_yticks(np.arange(-0.5, h, 1))
    ax.grid(which='both', color='lightgray', linewidth=0.5)

    # 可以隐藏刻度数字，只保留格线
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.set_xlabel("x (cols)")
    ax.set_ylabel("y (rows)")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.set_title("Grid Map (0=free, 1=obstacle)")

    plt.show()


if __name__ == "__main__":
    path = "grid_origin.npy"
    visualize_grid(path)
