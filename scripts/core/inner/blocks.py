# 实现基于BlockRegistry的CADMM blocks管理，包含BlockRegistry、pack/unpack、block slice、block residual等
# 1）对应文章中Algorithm 1 、 Algorithm 3 的变量快结构，包含如f_hat, B_hat, y_hat, sigma, r_hat等，将这些块结构在工程中变成可寻址的slices，并提供pack/unpack功能
# 2）对应文章中Algorithm 1 的停止准则，计算各个block的残差以及全局aggregated residual。
# 3）提供Residual balancing的block的统计支撑
"""
Target
- 给定BlockDef列表（有序），构造：
    * offsets[name] -> slice(start, end)
    * total_dim = D
- 提供pack/unpack
- 提供slice(name)以及names()
"""
import numpy as np

from scripts.core.data import BlockDef, FeatureFlags, BlockRegistry


# 基于flags构建registry
def make_registry(N:int, E:int, G:int, flags:FeatureFlags) -> BlockRegistry:
    """
    根据 flags.enable_* 决定包含哪些 blocks
    Output:
    - BlockRegistry
    """


# 按block视图获取slice
def get_block(reg: BlockRegistry, x: np.ndarray, name: str) -> np.ndarray:
    """返回 view"""

def set_block(reg: BlockRegistry, x: np.ndarray, name: str, value: np.ndarray) -> None:
    """把 value 写回 x 的 slice"""

# Stop criterion 需要的 eps 计算
def compute_eps_pri(eps_abs: float, eps_rel: float, q: np.ndarray, z: np.ndarray) -> float:
    """返回 epsilon_pri（全局）"""

def compute_eps_dual(eps_abs: float, eps_rel: float, u: np.ndarray, eta: float) -> float:
    """返回 epsilon_dual（全局）"""



# 计算block level的residuals
class ResidualModel:
    def primal_block_norms(self, q: np.ndarray, z: np.ndarray, reg: BlockRegistry) -> tuple[dict[str,float], float]:
        ...

    def dual_block_norms(self, z: np.ndarray, z_prev: np.ndarray, eta: float, reg: BlockRegistry) -> tuple[dict[str,float], float]:
        ...

# 共识B0实现
class ConsensusResidual(ResidualModel):
    pass
    # r_i = q_i - z
    # s   = eta*(z - z_prev)


# A/B 一般线性残差模型实现
class LinearResidual(ResidualModel):
    def __init__(self, A_list: list[np.ndarray], B: np.ndarray, c_list: list[np.ndarray]):
        ...
    # r_i = A_i q_i + B z - c_i
    # s   = eta*B.T@(z - z_prev)   (或你推导的版本)

def make_residual_model(problem, reg, flags) -> ResidualModel:
    pass


class BlockRegistry:
    def __init__(self, blocks: list[BlockDef]): ...
    @property
    def total_dim(self) -> int: ...
    def names(self) -> list[str]: ...
    def sl(self, name: str) -> slice: ...          # name -> slice
    def shape(self, name: str): ...
    def pack(self, blocks_dict: dict[str, np.ndarray]) -> np.ndarray: ...
    def unpack(self, x: np.ndarray) -> dict[str, np.ndarray]: ...

