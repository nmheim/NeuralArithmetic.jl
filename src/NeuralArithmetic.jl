module NeuralArithmetic

using Flux
using Zygote
using Flux: glorot_uniform, destructure

include("nau.jl")
include("nmu.jl")
include("nmux.jl")

include("nac.jl")
include("nalu.jl")
include("nalux.jl")

end # module
