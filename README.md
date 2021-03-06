# Semester-project-dynamical-system-learning
This the semester project accomplish during the period of fall 2019. The topic of the project consists into buiding a layer of neural network for times series forecasting. We test if the model can learn exsting dynamical systems, then we analyze its behaviour with respect to noise and finally, we use this layer in an architecture of neural net and apply it to a real time series.

## Libraries
Pytorch version 1.1.0

Pandas version 0.25.3

## Run the codes
There are three files for the code running:
- ``run_model_name.py``: Train the models on different dynamical systems and times series (take ``model_name=pendulum`` or ``model_name=lorenz``). Then Execute the command ``python path_files/run.py``.
- ``sim.py``: Proceed into an amount of simulation of the model on the generated data with gaussian noise and save the different features in a ``.csv`` -files. Execute the command ``python path_files/run.py``.

## File description
The content of this repository is composed of the following parts:

- Several py-files and two ipynb-file:
      
   -**``utils.py``**:
   
    - ``generate_pendumlum`` : generates the dynamical systems describing the pendulum motion.
    - ``generate_lorenz`` : generates the dynamical system describing the lorenz attractor.
    - ``tensor_norm`` : computes the tensor norm.
    - ``train_model``: computes the training loss of the model.
    - ``training_session``: computes the training loss and the optimal weights.
    - ``test_model``: computes the test loss of the model
    - ``create_matrix_time``: creates a tensor of multiple time series.
    - ``train_test_split`` : split the dataset into a train set and a test set.
    - ``input_target_split``: split the dataset into an input set and a target set.
    - ``run_session``: train the model.

    
   -**``DynanicalSystemLayer.py``**:
   
     - ``Class LinearODELayer``: Linear ode layer.
     - ``Class NonLinearODELayer``: Non-Linear ode layer.
     
     Each class contains three methods:
     
     - ``__init__``: constructs the class.
     - ``forward``: computes the predicted time series.
     - ``reset_parameters``: initializes the weights of the model.
     
   -**``models.py``**:
     
     - ``Class LinearODEModel`` : Linear ode model build with linear ode layer.
     - ``Class NNODEModel``: Quadratic ode model build with linear ode layer and non linear layer.
     - ``Class NeuralNetODEModel``: neural network containing two fully connected layers and the class ``NNODEMODEL``.
     - ``Class FullyLinearLayerModel``: neural network containing three connected layers.
     
     Each class contains two methods:
     
     - ``__init__``: constructs the class.
     - ``forward``: computes the predicted time series.
    
   -**``grid_search.py``**:
     - ``grid_search``: computes 40 simulation of the model and save the results according to selected parameters.
     - ``boxplot_feature``: plot features of a dataframe into a boxplot.
     
   -**``run_pendulum.py``**: run the model with pendulum.
   
   -**``run_lorenz.py``**: run the model with lorenz attractor.
   
   -**``sim.py``**: contains the procedure that generate the CSV-file containing the features from the simulations.
   
   -**``ode_demo.ipynb``**: contains the boxplots and the analysis of the model with noisy data.
   -**``times_series_forecast.ipynb``**: contains the training of the model on new times series and the benchmark of the model.
   
