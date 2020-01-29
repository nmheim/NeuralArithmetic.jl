module NeuralArithmetic

using Flux
using Zygote
using Flux: glorot_uniform
using LinearAlgebra: Diagonal

include("nac.jl")
include("nalu.jl")
include("nalux.jl")

include("nau.jl")
include("nmu.jl")

end # module
