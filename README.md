# Bayesian Neural Network
Translation of Bayesian Neural Network Model and its training algorithm to Julia (Chandra &amp; He (2021))

Link to the air quality: https://www.kaggle.com/datasets/decide-soluciones/air-quality-madrid

- data_prep.py is a script to prepare data to construct them into state-space
- neural_network.jl has all the helper function and should be ran first
- initialization.jl should be ran before main-loop and has all the hyper-parameters
- main_loop.jl is the Langevin-gradient MC and main loop
