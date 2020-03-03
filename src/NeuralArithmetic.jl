module NeuralArithmetic

using Flux
using Zygote
using Flux: glorot_uniform, destructure

include("nac.jl")
include("nalu.jl")
include("nalux.jl")

include("nau.jl")
include("nmu.jl")
include("nmux.jl")

end # module
