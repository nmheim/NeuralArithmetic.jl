using Test
using Flux
using LinearAlgebra
using NeuralArithmetic
using NeuralArithmetic: weights, gate
using ChainRulesTestUtils

include("nau.jl")
include("nmu.jl")
include("npu.jl")

include("nac.jl")
include("nalu.jl")
include("inalu.jl")

using DiffEqFlux
include("fast_nau.jl")
include("fast_npu.jl")
