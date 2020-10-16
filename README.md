# Neural Arithmetic

[![Build Status](https://travis-ci.com/nmheim/NeuralArithmetic.jl.svg?branch=master)](https://travis-ci.com/nmheim/NeuralArithmetic.jl)
[![codecov](https://codecov.io/gh/nmheim/NeuralArithmetic.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/nmheim/NeuralArithmetic.jl)

Collection of layers that can perform arithmetic operations such as addition,
subtraction, multiplication, and division in a single layer.  Implements
[NALU](https://arxiv.org/abs/1808.00508),
[iNALU](https://arxiv.org/abs/2003.07629),
[NMU & NAU](https://openreview.net/forum?id=H1gNOeHKPS), and [NPU](https://arxiv.org/abs/2006.01681).

And additionally `FastNAU` and `FastNPU` for use with [DiffEqFlux.jl](https://github.com/SciML/DiffEqFlux.jl).

# Simple Neural Arithmetic

As an example, we can train different layers to learn the function
```julia
f(x,y) = (x+y, x*y, 1/x, sqrt(x))
```
which has two inputs and four outputs. For mo.  The figure below plots the
error of each layer and arithmetic operation in `f` for training and testing
datapoints.  All layers were trained on the input range U(0.1,2). For more
details take a look at [our paper](https://arxiv.org/abs/2006.01681) and
the [code to reproduce](https://github.com/nmheim/NeuralPowerUnits) the image below.

![layers](img/layers.png)
