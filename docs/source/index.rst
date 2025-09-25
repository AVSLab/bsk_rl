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



**BSK-RL** (`Basilisk <https://hanspeterschaub.info/basilisk>`_ + 
`Reinforcement Learning <https://en.wikipedia.org/wiki/Reinforcement_learning>`_) is a 
Python package for constructing `Gymnasium <https://gymnasium.farama.org/index.html>`_ 
environments for spacecraft tasking problems. It is built on top of 
`Basilisk <https://hanspeterschaub.info/basilisk>`_, a modular and fast spacecraft 
simulation framework, making the simulation environments high-fidelity and computationally 
efficient. BSK-RL also includes a collection of utilities and examples 
for working with these environments.

A whitepaper on the design philosophy behind BSK-RL and an example use case can be 
:download:`downloaded here <_static/stephenson_bskrl_2024.pdf>`.

..  youtube:: 8qR-AGrCFQw

Quickstart
----------
Installation
^^^^^^^^^^^^
BSK-RL is available on PyPi and can be installed with pip. Simply run 

    .. code-block:: console

        $ pip install bsk-rl

Complete installation instructions for an editable installation and common troubleshooting tips can be found 
:doc:`here <install>`.


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
