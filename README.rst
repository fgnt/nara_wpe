========
nara_wpe
========

.. image:: https://readthedocs.org/projects/nara-wpe/badge/?version=latest
    :target: http://nara-wpe.readthedocs.io/en/latest/
    :alt: Documentation Status
    
.. image:: https://travis-ci.org/fgnt/nara_wpe.svg?branch=master
    :target: https://travis-ci.org/fgnt/nara_wpe
    :alt: Travis Status
    
.. image:: https://img.shields.io/pypi/v/nara-wpe.svg
    :target: https://pypi.org/project/nara-wpe/
    :alt: PyPI

.. image:: https://img.shields.io/pypi/dm/nara-wpe.svg
    :target: https://pypi.org/project/nara-wpe/
    :alt: PyPI

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://raw.githubusercontent.com/fgnt/nara_wpe/master/LICENSE
    :alt: MIT License


Weighted Prediction Error for speech dereverberation
====================================================

Background noise and signal reverberation due to reflections in an enclosure are the two main impairments in acoustic
signal processing and far-field speech recognition. This work addresses signal dereverberation techniques based on WPE for speech recognition and other far-field applications.
WPE is a compelling algorithm to blindly dereverberate acoustic signals based on long-term linear prediction.

The main algorithm is based on the following paper:
Yoshioka, Takuya, and Tomohiro Nakatani. "Generalization of multi-channel linear prediction methods for blind MIMO impulse response shortening." IEEE Transactions on Audio, Speech, and Language Processing 20.10 (2012): 2707-2720.


Content
=======

- Iterative offline WPE/ block-online WPE/ recursive frame-online WPE
- All algorithms implemented both in Numpy and in TensorFlow (works with version `1.12.0`).
- Continuously tested with Python 3.6, 3.7, 3.8 and 3.9.
- Automatically built documentation: `nara-wpe.readthedocs.io <https://nara-wpe.readthedocs.io/en/latest/>`_
- Modular design to facilitate changes for further research

Installation
============

Install it directly with Pip, if you just want to use it:

.. code-block:: bash

  pip install nara_wpe

If you want to make changes or want the most recent version: Clone the repository and install it as follows:

.. code-block:: bash

  git clone https://github.com/fgnt/nara_wpe.git
  cd nara_wpe
  pip install --editable .

Check the `example notebook <https://github.com/fgnt/nara_wpe/tree/master/examples>`_ for further details.
If you download the example notebook, you can listen to the input audio examples and to the dereverberated output too.


Citation
========

To cite this implementation, you can cite the following paper::

    @InProceedings{Drude2018NaraWPE,
      Title     = {{NARA-WPE}: A Python package for weighted prediction error dereverberation in {Numpy} and {Tensorflow} for online and offline processing},
      Author    = {Drude, Lukas and Heymann, Jahn and Boeddeker, Christoph and Haeb-Umbach, Reinhold},
      Booktitle = {13. ITG Fachtagung Sprachkommunikation (ITG 2018)},
      Year      = {2018},
      Month     = {Oct},
    }


To view the paper see
`IEEE Xplore <https://ieeexplore.ieee.org/document/8578026>`__ (`PDF <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8578026>`__)
or for a preview see `Paderborn University RIS <https://ris.uni-paderborn.de/publication/11873>`__ (`PDF <https://groups.uni-paderborn.de/nt/pubs/2018/ITG_2018_Drude_Paper.pdf>`__).



Comparision with the NTT WPE implementation
===========================================

The fairly recent John Hopkins University paper (Manohar, Vimal: `Acoustic Modeling for Overlapping Speech Recognition: JHU CHiME-5 Challenge System <https://ieeexplore.ieee.org/abstract/document/8682556>`_, ICASSP 2019) reporting on their CHiME 5 challenge results dedicate an entire table to the comparison of the Nara-WPE implementation and the NTT WPE implementation.
Their result is, that the Nara-WPE implementation is as least as good as the NTT WPE implementation in all their reported conditions.


Development history
====================

Since 2017-09-05 a TensorFlow implementation has been added to `nara_wpe`. It has been tested with a few test cases against the Numpy implementation.

The first version of the Numpy implementation was written in June 2017 while Lukas Drude and Kateřina Žmolíková resided in Nara, Japan. The aim was to have a publicly available implementation of Takuya Yoshioka's 2012 paper.
