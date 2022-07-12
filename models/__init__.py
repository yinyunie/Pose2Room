#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT

from . import p2rnet
from . import loss

method_paths = {
    'P2RNet': p2rnet
}

__all__ = ['method_paths']
