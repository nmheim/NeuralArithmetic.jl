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

@testset "show functions" begin
    layers = ["NPU","RealNPU","NaiveNPU","NAU","NMU","NAC","NALU",
              "FastNPU","FastRealNPU","FastNAU"]
    for l in layers
        m = eval(Symbol(l))(2,2)
        @test repr(m) == "$l(in=2, out=2)"
    end
end
