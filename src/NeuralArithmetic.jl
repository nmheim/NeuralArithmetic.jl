module NeuralArithmetic

using Flux
using Zygote
using Flux: glorot_uniform

include("nac.jl")
include("nalu.jl")

include("nau.jl")
include("nmu.jl")

end # module
