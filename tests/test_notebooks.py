import sys
import os
import subprocess
import tempfile
import unittest
import pytest

import nbformat

from nara_wpe import project_root


def _notebook_run(path):
    """Execute a notebook via nbconvert and collect output.
       :returns (parsed nb object, execution errors)
    """
    dirname = os.path.dirname(str(path))
    os.chdir(dirname)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=360",
                "--output", fout.name, str(path)]
        subprocess.check_call(args)

        fout.seek(0)
        nb = nbformat.read(fout, nbformat.current_nbformat)

    errors = [
        output for cell in nb.cells if "outputs" in cell
        for output in cell["outputs"]
        if output.output_type == "error"
    ]

    return nb, errors


root = project_root / 'examples'


@pytest.mark.skipif(sys.version_info >= (3, 6, 0), reason='Only with Python 3.6+')
def test_wpe_numpy_offline(self):
    nb, errors = _notebook_run(root / 'WPE_Numpy_offline.ipynb')
    assert errors == []


@pytest.mark.skipif(sys.version_info >= (3, 6, 0), reason='Only with Python 3.6+')
def test_wpe_numpy_online(self):
    nb, errors = _notebook_run(root / 'WPE_Numpy_online.ipynb')
    assert errors == []


@pytest.mark.skipif(not((3, 6, 0) <= sys.version_info < (3, 7, 0)), reason='Only with Python 3.6')
def test_wpe_tensorflow_offline():
    nb, errors = _notebook_run(root / 'WPE_Tensorflow_offline.ipynb')
    assert errors == []


@pytest.mark.skipif(not((3, 6, 0) <= sys.version_info < (3, 7, 0)), reason='Only with Python 3.6')
def test_wpe_tensorflow_online():
    nb, errors = _notebook_run(root / 'WPE_Tensorflow_online.ipynb')
    assert errors == []


# def test_NTT_wrapper(self):
#     nb, errors = _notebook_run(self.root / 'NTT_wrapper_offline.ipynb')
#     assert errors == []
