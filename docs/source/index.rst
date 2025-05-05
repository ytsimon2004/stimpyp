Welcome to Stimpyp documentation!
====================================

Stimpyp is designed for NERF ``Stimpy`` software output parsing, and other utilities.


Compatible Version
--------------------------

stimpy
^^^^^^^

- ``pyvstim``: `Bitbucket master branch <https://bitbucket.org/activision/pyvstim/src/master/>`_
- ``stimpy``: `Bitbucket master branch <https://bitbucket.org/activision/stimpy/src/master/>`_
- ``stimpy``: `Github master branch <https://github.com/vision-to-action/stimpy>`_


camera DAQ
^^^^^^^^^^^^^^

- ``labcams``: `Archived tagged version <https://github.com/ytsimon2004/labcams/tree/rig2_labcam_2109>`_, due to `Source version <https://bitbucket.org/jpcouto/labcams/src/master/>`_ still updating

- ``pycams`` `temp_release branch <https://bitbucket.org/activision/labcams/src/temp_release/>`_



Installation
------------

- Required python >= 3.10

.. code-block:: bash

    pip install stimpyp


Getting Started
---------------

.. toctree::
   :maxdepth: 4
   :caption: Examples

   get_start/index



API Reference
-----------------

.. toctree::
   :maxdepth: 1
   :caption: Modules

   api/stimpyp.parser.base
   api/stimpyp.parser.stimpy_core
   api/stimpyp.parser.stimpy_git
   api/stimpyp.parser.stimulus
   api/stimpyp.parser.camlog
   api/stimpyp.parser.preference
