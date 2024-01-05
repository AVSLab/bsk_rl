Installation
============

.. toctree::
    :maxdepth: 1


Instructions
------------
#. Install the `Basilisk <http://hanspeterschaub.info/basilisk/Install.html>`_ spacecraft simulation framework, following instructions for the appropriate operating system. Installation on MacOS and Linux is preferable to Windows. Use a Python virtual environment as suggested in the Basilisk installation instructions.
#. Clone the BSK-RL repository. 

   .. code-block:: console
        
       $ git clone git@github.com:AVSLab/bsk_rl.git

#. Move to the base directory of the repository.

   .. code-block:: console
        
       $ cd bsk_rl

#. Ensure that the virtual environment Basilisk is installed in is active. Install BSK-RL with the following command.

   .. code-block:: console

       (.venv) $ python -m pip install -e . && finish_install

   The first half of this command will install ``pip`` dependencies and an editable copy of the BSK-RL package. ``finish_install`` downloads data dependencies and other packages not available through ``pip``. The installation of Basilisk is also verified at this step.

#. Test the installation by running the example scripts from the base directory.

   .. code-block:: console

       (.venv) $ pytest tests/examples

   For additional verification, the unit tests and integration tests can also be executed.

   .. code-block:: console

       (.venv) $ pytest tests/unittest
       (.venv) $ pytest tests/integration


Common Issues
-------------

Please report new installation issues on GitHub.