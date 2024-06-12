BSK-RL: Environments for Spacecraft Planning and Scheduling
===========================================================

.. toctree::
    :hidden:

    install
    examples/index
    api_reference/index
    release_notes
    publications
    citation
    GitHub <https://github.com/AVSLab/bsk_rl/>


.. warning::
    
    The 1.0.0 release has significant changes from previous versions. See the 
    :doc:`Release Notes <release_notes>` for more information.


**BSK-RL** (`Basilisk <https://hanspeterschaub.info/basilisk>`_ + 
`Reinforcement Learning <https://en.wikipedia.org/wiki/Reinforcement_learning>`_) is a 
Python package for constructing `Gymnasium <https://gymnasium.farama.org/index.html>`_ 
environments for spacecraft tasking problems. It is built on top of 
`Basilisk <https://hanspeterschaub.info/basilisk>`_, a modular and fast spacecraft 
simulation framework, making the simulation environments high-fidelity and computationally 
efficient. BSK-RL also includes a collection of utilities and examples 
for working with these environments.

Quickstart
----------
Installation
^^^^^^^^^^^^
Complete installation instructions and common troubleshooting tips can be found 
:doc:`here <install>`. To install BSK-RL:

#. Install the `Basilisk <https://hanspeterschaub.info/basilisk>`_ spacecraft simulation 
   framework.
#. Clone BSK-RL.

    .. code-block:: console

        $ git clone git@github.com:AVSLab/bsk_rl.git && cd bsk_rl

#. Install BSK-RL in the same virtual environment as Basilisk.

    .. code-block:: console

        (.venv) $ python -m pip install -e . && finish_install

#. Test the installation.

    .. code-block:: console

        (.venv) $ pytest .

Construct an Environment
^^^^^^^^^^^^^^^^^^^^^^^^

A quick but comprehensive tutorial can be found at :doc:`examples/simple_environment`.


Acknowledgements
----------------
BSK-RL is developed by the `Autonomous Vehicle Systems (AVS) Lab <https://hanspeterschaub.info/AVSlab.html>`_ 
at the University of Colorado Boulder. The AVS Lab is part of the `Colorado Center for Astrodynamics Research (CCAR) <https://www.colorado.edu/ccar>`_ 
and the `Department of Aerospace Engineering Sciences <https://www.colorado.edu/aerospace/>`_.

Development has been supported by NASA Space Technology Graduate Research Opportunity 
(NSTGRO) grants, 80NSSC20K1162 and 80NSSC23K1182. This work has also been supported by 
Air Force Research Lab grant FA9453-22-2-0050. 

Development of this software has utilized the Alpine high performance computing resource
at the University of Colorado Boulder. Alpine is jointly funded by the University of
Colorado Boulder, the University of Colorado Anschutz, and Colorado State University.