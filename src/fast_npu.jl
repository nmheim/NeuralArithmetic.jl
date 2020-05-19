using NeuralArithmetic
using Zygote
using Flux
using ForwardDiff

struct FastGatedNPUX{F} <: DiffEqFlux.FastLayer
    in::Int
    out::Int
    initial_params::F
    function FastGatedNPUX(in::Int, out::Int;
                           initRe=Flux.glorot_uniform, initIm=Flux.glorot_uniform)
        initial_params() = vcat(vec(initRe(out,in)),
                                vec(initIm(out,in)),
                                Flux.ones(in)/2)
        new{typeof(initial_params)}(in, out, initial_params)
    end
end

DiffEqFlux.paramlength(f::FastGatedNPUX) = (f.out*f.in)*2 + f.in
DiffEqFlux.initial_params(f::FastGatedNPUX) = f.initial_params()

function _restructure(f::FastGatedNPUX, p::AbstractVector)
    len = f.out * f.in
    Re = reshape(p[1:len], f.out, f.in)
    Im = reshape(p[(len+1):(2*len)], f.out, f.in)
    g  = p[(2*len+1):end]
    return (Re,Im,g)
end

#_restructure(f::FastGatedNPUX, p::AbstractVector, nonfast=true) = GatedNPUX(_restructure(p)...)

function (f::FastGatedNPUX)(x::AbstractVector,p::AbstractVector)
    (Re,Im,g) = _restructure(f,p)
    NeuralArithmetic.mult(Re,Im,g,x)
end

function (f::FastGatedNPUX)(x::AbstractMatrix,p::AbstractVector)
    (Re,Im,g) = _restructure(f,p)
    NeuralArithmetic.mult(Re,Im,g,x)
end

Zygote.@adjoint function (f::FastGatedNPUX)(x::AbstractVector,p::AbstractVector)
    lenx = length(x)
    function _f(z::Vector)
        x = reshape(z[1:lenx], lenx)
        p = z[(lenx+1):end]
        f(x,p)
    end

    Jz = ForwardDiff.jacobian(_f, vcat(x, p))
    Jx = Jz[:,1:lenx]
    Jp = Jz[:,(lenx+1):end]
    f(x,p), Δ -> (@show size(Δ); (nothing, (Δ'*Jx)', (Δ'*Jp)'))
end

# Zygote.@adjoint function (f::FastGatedNPUX)(x::AbstractMatrix,p::AbstractVector)
#     display("matrix ajdoint")
#     #lenx = size(x,1)
#     #function _f(z::Vector)
#     #    x = reshape(z[1:lenx], lenx)
#     #    p = z[(lenx+1):end]
#     #    f(x,p)
#     #end
# 
#     lenx = length(x)
#     sizx = size(x)
#     function _f(z::Vector)
#         x = reshape(z[1:lenx], sizx)
#         p = z[(lenx+1):end]
#         vec(f(x,p))
#     end
# 
#     @show size(x)
#     @show size(_f(vcat(vec(x),p)))
#     #Jz = map(c -> ForwardDiff.jacobian(_f,vcat(c,p)), eachcol(x))
#     Jz = ForwardDiff.jacobian(_f,vcat(vec(x),p))
#     @show size(Jz)
#     # Jx = reshape(Jz[:,1:lenx], sizx)
#     # Jp = vec(Jz[:,(lenx+1):end])
#     # @show size(Jx)
#     # @show size(Jp)
# 
#     f(x,p), Δ -> (@show size(Δ); (nothing,nothing,nothing))
# end
   

#(dec::FluxODEDecoder)(Z::AbstractMatrix) = hcat([dec(Z[:,ii]) for ii in 1:size(Z,2)]...)

# using Random
# Random.seed!(0)
# m = FastGatedNPUX(2,3)
# x = rand(2)
# p = rand(paramlength(m))
# loss(x) = sum(m(x,p))
# @show Flux.gradient(loss,x)[1]

# x = rand(2,6)
# p = rand(paramlength(m))
# loss(x) = sum(m(x,p))
# Flux.gradient(loss,x)[1]
# @show "asdf"
