# Semester-project-dynamical-system-learning
This the semester project accomplish during the period of fall 2019. The topic of the project consists into buiding a layer of neural network for times series forecasting. We test if the model can learn exsting dynamical systems, then we analyze its behaviour with respect to noise.

## Libraries
Pytorch version 1.1.0

Pandas version 0.25.3

## Run the codes
There are three files for the code running:
- ``run.py``: Train the models on different dynamical systems and times series (take ``model_name=pend`` or ``model_name=lorenz``).
- ``sim.py``: Proceed into an amount of simulation of the model on the generated data with gaussian noise and save the different features in a ``.csv`` -files.

## File description
The content of this project is composed of the following parts:
      - The folder ``Semester-project-dynamical-system-learning`` that contains several py-files and one ipynb-file
            - **``utils.py``**:
                 - ``generate_pendumlum`` : generate the dynamical systems describing the pendulum motion.
                 - ``generate_lorenz`` : generate the dynamical system describing the lorenz attractor.
