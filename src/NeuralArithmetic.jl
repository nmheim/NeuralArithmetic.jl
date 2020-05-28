module NeuralArithmetic

using Flux
using Zygote
using Flux: glorot_uniform, destructure
using Requires

include("nau.jl")
include("nmu.jl")
include("npu.jl")

function __init__()
    @require DiffEqFlux="aae7a2af-3d4f-5e19-a356-7da93b79d9d0" include("fast_nau.jl")
    @require DiffEqFlux="aae7a2af-3d4f-5e19-a356-7da93b79d9d0" include("fast_npu.jl")
end

include("nac.jl")
include("nalu.jl")
include("inalu.jl")

end # module
