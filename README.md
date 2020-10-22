# Neural Arithmetic

[![Build Status](https://travis-ci.com/nmheim/NeuralArithmetic.jl.svg?branch=master)](https://travis-ci.com/nmheim/NeuralArithmetic.jl)
[![codecov](https://codecov.io/gh/nmheim/NeuralArithmetic.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/nmheim/NeuralArithmetic.jl)

Collection of layers that can perform arithmetic operations such as addition,
subtraction, multiplication, and division in a single layer.  Implements
[NPU](https://arxiv.org/abs/2006.01681),
[NMU & NAU](https://openreview.net/forum?id=H1gNOeHKPS)
[NALU](https://arxiv.org/abs/1808.00508), and
[iNALU](https://arxiv.org/abs/2003.07629).

And additionally `FastNAU` and `FastNPU` for use with [DiffEqFlux.jl](https://github.com/SciML/DiffEqFlux.jl).

# A Simple Example

The script [examples/npu.jl](examples/npu.jl) illustrates how to learn a the function `f`
```julia
f(x,y) = (x+y, x*y, 1/x, sqrt(x))
```
with a stack of NPU and NAU.
The NPU can perform multiplication, division, and other power functions of its inputs.
An NPU with two inputs `x` and `y` can perform `x^a * y^b` for each hidden variable.
The NAU is just a matmul, so it can perform `a*x + b*y` (i.e. addition/subtraction).
For more information check out the [NPU paper](https://arxiv.org/abs/2006.01681).

The image below shows the learned weights of the model compared to the perfect solution.
The first plot shows the real weights of the NPU, where the first
row forms the first hidden activation `h1 = x^1*y^1 = x*y`, the second row
forms the second hidden activation `h2 = x^1*y^0`, and so on.
The last row performs division `h6 = x^0*y^(-1)`.
The NAU performs the remaining addition in the first row.

![npu](img/npu_example.png)



# Comparing Neural Arithmetic Units

As an example, we can train different layers to learn the function
```julia
f(x,y) = (x+y, x*y, x/y, sqrt(x))
```
which has two inputs and four outputs. For mo.  The figure below plots the
error of each layer and arithmetic operation in `f` for training and testing
datapoints.  All layers were trained on the input range U(0.1,2). For more
details take a look at [our paper](https://arxiv.org/abs/2006.01681) and
the [code to reproduce](https://github.com/nmheim/NeuralPowerUnits) the image below.

![layers](img/layers.png)
