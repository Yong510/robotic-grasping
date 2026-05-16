# -*- coding: utf-8 -*-
"""
模型注册模块
训练脚本通过 get_network 函数获取对应的模型类
"""

def get_network(network_name):
    """
    根据网络名称返回对应的模型类（不是实例）
    """
    network_name = network_name.lower()
    if network_name == 'grconvnet':
        from .grconvnet import GenerativeResnet
        return GenerativeResnet
    elif network_name == 'grconvnet2':
        from .grconvnet2 import GenerativeResnet
        return GenerativeResnet
    elif network_name == 'grconvnet3':
        from .grconvnet3 import GenerativeResnet
        return GenerativeResnet
    elif network_name == 'grconvnet4':
        from .grconvnet4 import GenerativeResnet
        return GenerativeResnet
    # ========== 新增：MFENet 分支 ==========
    elif network_name == 'mfenet':
        from .mfenet import MFENet
        return MFENet
    # =====================================
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))

# 为了保持向后兼容，显式导出所有模型类（可选）
from .grconvnet import GenerativeResnet
from .grconvnet2 import GenerativeResnet as GenerativeResnet2
from .grconvnet3 import GenerativeResnet as GenerativeResnet3
from .grconvnet4 import GenerativeResnet as GenerativeResnet4
from .mfenet import MFENet  # 新增这一行