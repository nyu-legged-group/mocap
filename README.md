# Motion Capture Project
This is a motion capture project transferred from human to humanoid
# 
All the instructions are given in the notebooks/throw.ipynb

# Installation instructions

You can run the Jupyter notebook for the project using a local installation of Python on your computer.

## With a local Python installation:

You will need to have a local installation of Python (3.7 to 3.9 recommended). We recommend that you install Python using [Anaconda](https://www.anaconda.com/products/individual)

You then will need to install 
1. [Pinocchio](https://github.com/stack-of-tasks/pinocchio), which is a rigid body dynamics library
2. [Meshcat](https://github.com/rdeits/meshcat-python), which is a convenient visualizer
3. [humanoid_property](https://github.com/ddliu365/humanoid_property)(devel branch), which contains the URDF file for our humanoid. For humanoid_property installation, you can run the following:
``python setup.py install --user`` or ``python3 setup.py install --user`` if you are using python3.

With an Anaconda installation, you can run the following:
``conda install pinocchio meshcat-python``

Your environment should then be all set to run the Notebook.
