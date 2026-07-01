This is a repo that 
    1. stores different parameter estimation methods, 
    2. Generate datasets of varying qualities (how non-linear, size of the dataset)
    3. Apply these methods to test their performance, and also illustrate their pros and cons

Currently, we are working on Point 2. Generate the data.

We want to have many different functions that generate the data, and here are a few concerns in how the data are generated
1. The numbers of parameters (dim of inputs), outputs, and how many data there are
2. How linear/non-linear the data are
3. Whether the data are from some relatively more sophisticated models, e.g., from PDEs
4. Whether we should consider observational error (more later)


And the different functions should output the following:
1. The X, the Y, the true observations, and the correponding parameters that are used to generate the data.
    Here we should allow for 10-100 parameters, 30-1000 data points, and 10-500 observations
2. They should all be stored in one file
3. We should be able to control what observatoins have structural error, and the level of that structural error

So currently the challenge is how flexible we make the function? To what level we define them? 
The other equally important question is that we must be able to visualize the data based on how they are generated.
In my ideal world, I want something that's like a chart that has bars showing (not exactly but rougly) how non-linear they are, how sparse they are and etc. 

We also need to be able to visualize the dataset. 