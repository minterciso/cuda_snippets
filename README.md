#Introduction

In this repository you'll find some snippets of common CUDA operations. I'll always try to make them as pure C/C++ as possible (ie: trying to avoid using STL or 3rd party libraries).

##curand_dev
Shows how to create a lot of numbers on the device, then add them on some pre-specified "bins". This is usefull to check the probability of hitting some number.
I focused on 2 main distributions: Uniform and Normal

###Compilation
Use nvcc and just add the -lcurand library:
    $ nvcc -o curand_dev curand_dev.cu -lcurand

###Usage
    Usage: ./curand_dev options
            -h              --help                  This help message
            -d type         --distribution type     Distribution type (valid values: 'uniform', 'normal')
            -s value        --scale value           Scale value for the Normal distribution.
            -S value        --shift value           Shift value for the Normal distribution.
            -m value        --min value             Minimum value
            -M value        --max value             Maximum value
            -o file         --output file           Output CSV file (required)
            -g              --gnuplot               Write a gnuplot file for showing the output data

####Distribution type
There are 2 main distribution presented here: uniform and normal.

The Uniform distribution simply generates a number using a uniform distribution, and *caps* the number between [min,max[. The amount of capped numbers is the probability of each number to show using this distribution. The graph of the probabilities should be pretty standard.

The Normal distribution on the other hand returns a number between [0,1.0[, and we then scale it to an int between [min,max[. The normal distribution uses mean 0.0 and standar deviation of 1.0. It's also possible to adjus the *scale* (or spread) of the probabilities and the *shift* of the distribution via the parameters -s and -S. For instance:
    ./curand_dev -o output.csv -d normal -s 10 -S 50
Would give numbers between [0,100[ with a scale of 10 and shift of 50 would result in a graph that is centered on the middle and spreads values trough all bins. If for instace we change the -s value, the spread size would be much smaller. I recommend trying it.

There's also a thir hidden option that, if you pass anything else then "normal" or "uniform", it'll try to create both graphs, and output the result of both on the same file. This is VERY usefull for seeing the differences between the distributions.

