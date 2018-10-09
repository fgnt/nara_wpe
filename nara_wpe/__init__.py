try:
    import pathlib
except ImportError:
    # Python 2.7
    import pathlib2 as pathlib

import os

project_root = pathlib.Path(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
))

name = "nara_wpe"