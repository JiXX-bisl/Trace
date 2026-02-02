# 通信窗口 W + Omega 冻结，作为异步/时变图基础，支持TTL信息过滤
from scripts.core.data import LinkSnapshot, CommWindowState


def open_new_window(step: int, link: LinkSnapshot, W: int, rng) -> CommWindowState:
    ...


def get_window_link(step: int, window: CommWindowState, link_current: LinkSnapshot, flags) -> LinkSnapshot:
    ...

