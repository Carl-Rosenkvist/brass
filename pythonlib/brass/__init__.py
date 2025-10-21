from . import _brass            # bind the module object
from ._brass import *           # re-export its names

from .scan.template import smash_cmd
from .scan.scan import Scan
from .hist2d import Hist2D
from .hist1d import Hist1D
from .meta.meta import MetaBuilder

import atexit
if hasattr(_brass, "_clear_registry"):
    atexit.register(_brass._clear_registry)

__all__ = [name for name in dir() if not name.startswith("_")]
