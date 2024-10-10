Installation
============

.. toctree::
    :maxdepth: 1


Instructions
------------
#. Install the `Basilisk <http://hanspeterschaub.info/basilisk/Install.html>`_ spacecraft
   simulation framework, following instructions for the appropriate operating system.
   Installation on MacOS and Linux is preferable to Windows. Use a Python virtual
   environment as suggested in the Basilisk installation instructions.
#. Clone the BSK-RL repository. 

   .. code-block:: console
        
       $ git clone git@github.com:AVSLab/bsk_rl.git

#. Move to the base directory of the repository.

   .. code-block:: console
        
       $ cd bsk_rl

#. Ensure that the virtual environment Basilisk is installed in is active. Install
   BSK-RL with the following command.

   .. code-block:: console

       (.venv) $ python -m pip install -e "." && finish_install

   The first half of this command will install ``pip`` dependencies and an editable copy
   of the BSK-RL package. ``finish_install`` downloads data dependencies and verifies the
   installation of Basilisk.

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

SPICE Errors
^^^^^^^^^^^^

Errors such as 

   .. code-block:: console

        Toolkit version: N0065

        SPICE(NOSUCHFILE) --

        The attempt to load
        "/home/user/basilisk/dist3/Basilisk/supportData/EphemerisData/de430.bsp" by
        the routine FURNSH failed. It could not be located.

        A traceback follows. The name of the highest level module is first.
        furnsh_c --> FURNSH --> ZZLDKER

can be resolved by ensuring that `Basilisk is installed using git-lfs <http://hanspeterschaub.info/basilisk/Install/pullCloneBSK.html>`_.
