import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class ObstacleBox:
    # 连续空间中的障碍物边界（单位：米）
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


def create_random_3d_space(
    size_m=(40.0, 40.0, 8.0),      # 空间尺寸 (X, Y, Z) 米
    resolution=0.5,                # 体素分辨率（米/格）
    num_obstacles=30,              # 随机障碍物数量
    obstacle_size_range=((1.0, 6.0), (1.0, 6.0), (1.0, 4.0)),  # 障碍物尺寸范围（x/y/z）
    seed=42,
    allow_overlap=False,           # 是否允许障碍物重叠
    ground_attached=True           # 障碍物是否从地面开始（z=0）
):
    """
    返回：
      occupancy: np.ndarray, shape=(nx, ny, nz), uint8
                 0=自由空间, 1=障碍物
      obstacles: List[ObstacleBox]
      meta: dict (空间信息)
    """
    rng = np.random.default_rng(seed)

    Lx, Ly, Lz = size_m
    nx = int(np.round(Lx / resolution))
    ny = int(np.round(Ly / resolution))
    nz = int(np.round(Lz / resolution))

    occupancy = np.zeros((nx, ny, nz), dtype=np.uint8)
    obstacles = []

    def meters_to_voxels(length_m):
        return max(1, int(np.round(length_m / resolution)))

    max_trials = num_obstacles * 50
    placed = 0
    trials = 0

    while placed < num_obstacles and trials < max_trials:
        trials += 1

        # 随机障碍物尺寸（米）
        sx_m = rng.uniform(*obstacle_size_range[0])
        sy_m = rng.uniform(*obstacle_size_range[1])
        sz_m = rng.uniform(*obstacle_size_range[2])

        sx = meters_to_voxels(sx_m)
        sy = meters_to_voxels(sy_m)
        sz = meters_to_voxels(sz_m)

        # 尺寸不能超过空间
        if sx >= nx or sy >= ny or sz >= nz:
            continue

        # 随机放置位置（体素索引）
        ix0 = rng.integers(0, nx - sx)
        iy0 = rng.integers(0, ny - sy)
        if ground_attached:
            iz0 = 0
        else:
            iz0 = rng.integers(0, nz - sz)

        ix1, iy1, iz1 = ix0 + sx, iy0 + sy, iz0 + sz

        # 检查重叠
        if not allow_overlap:
            if np.any(occupancy[ix0:ix1, iy0:iy1, iz0:iz1] == 1):
                continue

        # 写入障碍物
        occupancy[ix0:ix1, iy0:iy1, iz0:iz1] = 1

        # 转回连续空间（米）
        obstacles.append(
            ObstacleBox(
                x_min=ix0 * resolution,
                x_max=ix1 * resolution,
                y_min=iy0 * resolution,
                y_max=iy1 * resolution,
                z_min=iz0 * resolution,
                z_max=iz1 * resolution,
            )
        )
        placed += 1

    meta = {
        "size_m": size_m,
        "resolution": resolution,
        "grid_shape": occupancy.shape,
        "num_obstacles_requested": num_obstacles,
        "num_obstacles_placed": placed,
        "seed": seed,
    }
    return occupancy, obstacles, meta


def visualize_voxel_space(occupancy, resolution=0.5, max_voxels_for_plot=120000):
    """
    使用 matplotlib 显示体素障碍物（体素太多时会比较慢）
    """
    occ = occupancy.astype(bool)

    # 如果体素太密，可以简单下采样显示
    n_vox = occ.size
    step = 1
    if n_vox > max_voxels_for_plot:
        step = int(np.ceil((n_vox / max_voxels_for_plot) ** (1/3)))

    occ_show = occ[::step, ::step, ::step]
    res_show = resolution * step

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(occ_show, edgecolor="k", linewidth=0.1)

    nx, ny, nz = occ_show.shape
    ax.set_xlabel(f"X ({res_show:.2f} m/voxel)")
    ax.set_ylabel(f"Y ({res_show:.2f} m/voxel)")
    ax.set_zlabel(f"Z ({res_show:.2f} m/voxel)")
    ax.set_title("Random Obstacles in 40m x 40m x 8m 3D Space")
    ax.set_box_aspect((nx, ny, nz))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    occupancy, obstacles, meta = create_random_3d_space(
        size_m=(40.0, 40.0, 8.0),
        resolution=0.5,       # 0.5m 体素
        num_obstacles=35,
        obstacle_size_range=((1.0, 5.0), (1.0, 5.0), (1.0, 6.0)),
        seed=123,
        allow_overlap=False,
        ground_attached=True
    )

    print("Meta:", meta)
    print("前5个障碍物（米）:")
    for ob in obstacles[:5]:
        print(ob)

    # 保存体素地图（可用于后续算法）
    np.save("occupancy_40x40x8.npy", occupancy)

    # 可视化
    visualize_voxel_space(occupancy, resolution=meta["resolution"])