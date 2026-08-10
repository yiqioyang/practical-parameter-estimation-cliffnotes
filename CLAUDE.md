This is a repo that 
    1. stores different parameter estimation methods, 
    2. Generate datasets of varying qualities (how non-linear, size of the dataset)
    3. Apply these methods to test their performance, and also illustrate their pros and cons



2026 July 17th
Currently, we are working on Point 2. Generate the data. We are skipping Points 1 and 3 for now
One important concern is that we need to generate the data systematically across different kind of data (linear, non-linear, PDE and etc)
We want to highlight the full (if possible) spectrum of challenges in practical parameter estimation.
A few challenges are listed here:
    a) Parameter estimation efficiency (e.g., the use of more efficient sampling strategy);
    b) Poor emulator performance;
    c) Structural error;
    d) The strategy if we are given too few (hard to constrian the parameters effectively) or too many targets (structural error-prone); 
    e) When we do waves/iterations of parameter estimation, what is the best strategy (whether take all previous runs in training or a mixture, etc);
    f) Using scores or raw model outputs as emulator targets;
    g) If sensitivity test could help inform more efficient sampling strategy? 
    h) The presence of multi-modes in the solutions.
    i) The relative importance of parameters (some parameters are important after others are better constrained)

Another challenge is to make all the generated data formated systematically. For example, a nc file that has everything in it or 
csv files and etc. 

Similarly, once we have an ensemble of estimated parameters, we also need to be able to apply them to the moedel and check the results. This 
will help us deploy the methodology of simulation (which is what is this for)-emulation-sampling iteration.

It is also good to be able to inject different kind of errors or uncertainties to the data (e.g., obs error, structural error). Here the structural error can be something like a set of x_1 values that map to y_1 given a model m_1, in addition, we are also given some y_2 that is generated from a different x_2 given model y_2. x_1 and x_2 share the same parameter space. This set up is to simulate the scenario where there is no parameter values that could satisify y_1 and y_2 given m_1 and m_2

