========
nara_wpe
========

.. image:: https://readthedocs.org/projects/nara_wpe/badge/?version=pypi-release
    :target: http://nara-wpe.readthedocs.io/en/pypi-release/
    :alt: Documentation Status

.. image:: https://travis-ci.org/fgnt/nara_wpe.svg?branch=master
    :target: https://travis-ci.org/fgnt/nara_wpe

Different implementations of "Weighted Prediction Error" for speech dereverberation:
====================================================================================

Yoshioka, Takuya, and Tomohiro Nakatani. "Generalization of multi-channel linear prediction methods for blind MIMO impulse response shortening." IEEE Transactions on Audio, Speech, and Language Processing 20.10 (2012): 2707-2720.

This code has been tested with Python 3.6.

Clone the repository. Then install it as follows:

.. code-block:: bash

  https://github.com/fgnt/nara_wpe.git
  cd nara_wpe
  pip install --user -e .

Alternatively, install it directly with Pip from Github:

.. code-block:: bash

  pip install git+https://github.com/fgnt/nara_wpe.git

Check the example notebook for further details.
If you download the example notebook, you can listen to the audio examples of input and dereverberated output, too.

You can find some documentation here:
`nara-wpe.readthedocs.io 
<https://nara-wpe.readthedocs.io/en/latest/>`_.

Development history:
====================

Since 2017-09-05 a TensorFlow implementation is added to `nara_wpe`. It is tested with a few test cases against the Numpy implementation.

The first version of the Numpy implementation was written in June 2017 while Lukas Drude and Kateřina Žmolíková resided in Nara, Japan. The aim was to have a publicaly available implementation of Takuya Yoshioka's 2012 paper.
