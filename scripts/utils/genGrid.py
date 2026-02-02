"""
(x, y) = (列索引, 行索引)，原点在左下角，单位是“格”。
"""
import numpy as np
import matplotlib.pyplot as plt

GRID_SIZE = 80  # 40x40 网格

# 初始化地图：0 = 自由，1 = 障碍
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

# 撤回用的历史记录：每次笔画是一组 (row, col) 的列表
history = []

# 当前是否在绘制（按住左键拖动）
is_drawing = False
current_stroke = set()  # 本次笔画中被修改过的单元格


def get_cell_from_event(event, ax):
    """根据鼠标事件获取所在的格子坐标 (row, col)，不在范围内返回 None。"""
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return None

    # imshow(origin='lower') 情况下，x 对应列，y 对应行
    col = int(event.xdata)
    row = int(event.ydata)

    if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
        return row, col
    else:
        return None


def paint_cell(row, col, im, fig):
    """把指定格子涂成障碍物（1），并刷新显示。"""
    global grid, current_stroke

    # 只记录从 0 -> 1 的变化，这样撤回才好恢复
    if grid[row, col] == 0:
        grid[row, col] = 1
        current_stroke.add((row, col))
        im.set_data(grid)
        fig.canvas.draw_idle()


def on_button_press(event, ax, im, fig):
    """鼠标按下事件：左键开始绘制。"""
    global is_drawing, current_stroke

    if event.button == 1:  # 左键
        is_drawing = True
        current_stroke = set()
        cell = get_cell_from_event(event, ax)
        if cell is not None:
            row, col = cell
            paint_cell(row, col, im, fig)


def on_button_release(event, ax, im, fig):
    """鼠标松开事件：结束一次笔画并加入历史。"""
    global is_drawing, current_stroke, history

    if event.button == 1 and is_drawing:
        is_drawing = False
        if current_stroke:
            # 把本次笔画记录进历史，用于撤回
            history.append(list(current_stroke))
        current_stroke = set()


def on_motion(event, ax, im, fig):
    """鼠标移动事件：按住左键时连续绘制。"""
    global is_drawing

    if not is_drawing:
        return

    cell = get_cell_from_event(event, ax)
    if cell is not None:
        row, col = cell
        paint_cell(row, col, im, fig)


def undo_last_stroke(im, fig):
    """撤回上一笔：把该笔画中涂黑的格子还原为 0。"""
    global grid, history

    if not history:
        print("没有可以撤回的操作。")
        return

    last_stroke = history.pop()
    for (row, col) in last_stroke:
        grid[row, col] = 0

    im.set_data(grid)
    fig.canvas.draw_idle()
    print("撤回了一次绘制。")


def save_grid():
    """保存当前 grid 为 .npy 文件到指定路径。"""
    import os
    path = "grid_origin.npy"
    np.save(path, grid)
    print(f"地图已保存到: {os.path.abspath(path)}")


def on_key_press(event, fig, im):
    """键盘事件：z 撤回, s 保存, q 退出。"""
    if event.key == 'z':
        undo_last_stroke(im, fig)
    elif event.key == 'a':
        save_grid()
    elif event.key == 'q':
        print("退出编辑器。")
        plt.close(fig)


def main():
    global grid

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("40x40 栅格地图编辑器")

    # 显示栅格：0 白色，1 黑色
    im = ax.imshow(grid, cmap='gray_r', vmin=0, vmax=1, origin='lower')

    # 设置网格线
    ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1))
    ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1))
    ax.grid(which='both', color='lightgray', linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)

    # 绑定事件
    fig.canvas.mpl_connect(
        'button_press_event',
        lambda e: on_button_press(e, ax, im, fig)
    )
    fig.canvas.mpl_connect(
        'button_release_event',
        lambda e: on_button_release(e, ax, im, fig)
    )
    fig.canvas.mpl_connect(
        'motion_notify_event',
        lambda e: on_motion(e, ax, im, fig)
    )
    fig.canvas.mpl_connect(
        'key_press_event',
        lambda e: on_key_press(e, fig, im)
    )

    print("操作说明：")
    print("  左键点击/拖动：绘制障碍物（黑色，值=1）")
    print("  按键 z：撤回上一笔绘制")
    print("  按键 s：保存当前地图为 .npy 文件")
    print("  按键 q：退出编辑器")

    plt.show()


if __name__ == "__main__":
    main()
