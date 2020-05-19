module NeuralArithmetic

using Flux
using Zygote
using Flux: glorot_uniform, destructure
using DiffEqFlux

include("nau.jl")
include("nmu.jl")
include("npu.jl")

include("nac.jl")
include("nalu.jl")
include("nalux.jl")

include("fast_nau.jl")
include("fast_npu.jl")

end # module
