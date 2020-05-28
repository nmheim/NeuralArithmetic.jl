using Test
using Flux
using DiffEqFlux
using LinearAlgebra
using NeuralArithmetic
using NeuralArithmetic: weights, gate

include("nau.jl")
include("nmu.jl")
include("npu.jl")

include("fast_nau.jl")
include("fast_npu.jl")

include("nac.jl")
include("nalu.jl")
include("inalu.jl")
