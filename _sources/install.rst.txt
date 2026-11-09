Installation
============

.. toctree::
    :maxdepth: 1


Quick Installation
------------------
BSK-RL is available on PyPi and can be installed with pip. Simply run

      .. code-block:: console
   
         $ pip install bsk-rl


Editable Installation
---------------------
#. Install the `Basilisk <http://hanspeterschaub.info/basilisk/Install.html>`_ spacecraft
   simulation framework, either with `pip install bsk` or by following instructions for
   the appropriate operating system. Installation on MacOS and Linux are preferable to
   Windows. Use a Python virtual environment as suggested in the Basilisk installation
   instructions.
#. Clone the BSK-RL repository over SSH:

   .. code-block:: console
        
       $ git clone git@github.com:AVSLab/bsk_rl.git

   or over HTTPS, as some networks block SSH:

   .. code-block:: console

       $ git clone https://github.com/AVSLab/bsk_rl.git

#. Move to the base directory of the repository.

   .. code-block:: console
        
       $ cd bsk_rl

#. Ensure that the virtual environment Basilisk is installed in is active. Install
   BSK-RL with the following command.

   .. code-block:: console

       (.venv) $ python -m pip install -e "."

   The first half of this command will install ``pip`` dependencies and an editable copy
   of the BSK-RL package.

   For a more granular installation, ``.[docs]`` (for documentation dependencies) or 
   ``.[rllib]`` (for RLlib tools) can be specified. ``.[all]`` installs all dependencies.

#. Test the installation by running the unit tests and integration tests.

   .. code-block:: console

       (.venv) $ pytest tests/unittest
       (.venv) $ pytest tests/integration

   The installation can also be verified by running :doc:`examples/index` from the ``examples``
   directory.

#. To build documentation locally, run:

   .. code-block:: console

       (.venv) $ cd docs
       (.venv) $ make html
       (.venv) $ make view


Common Issues
-------------

Please report new installation issues on GitHub.