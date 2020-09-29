export NaiveNPU, RealNaiveNPU, RealNPU, NPU

"""
  NPU(in::Int, out::Int; initRe=glorot_uniform, initIm=Flux.zeros, initg=init05)

Neural Power Unit that can learn arbitrary power functions by using a complex
weights. Uses gating on inputs to simplify learning. In 1D the layer looks
like:

    g = min(max(g, 0), 1)
    r = abs(x) + eps(T)
    r = g*r + (1-g)*T(1)
    k = r < 0 ? pi : 0.0
    exp(Re*log(r) - Im*k) * cos(Re*k + Im*log(r))
"""
struct NPU{M<:AbstractMatrix,V<:AbstractVector}
    Re::M
    Im::M
    g::V
end

function NPU(in::Int, out::Int;
    initRe=glorot_uniform, initIm=Flux.zeros, initg=init05)
    Re = initRe(out, in)
    Im = initIm(out, in)
    g  = initg(in)
    NPU(Re,Im,g)
end

Flux.@functor NPU

function mult(Re::AbstractMatrix{T}, Im::AbstractMatrix{T}, g::AbstractVector{T}, x::AbstractArray{T}) where T
    g = gateclip(g)

    r = abs.(x) .+ eps(T)
    r = g .* r .+ (1 .- g)

    k = max.(-sign.(x), 0) .* T(pi)
    k = g .* k

    exp.(Re*log.(r) - Im*k) .* cos.(Re*k + Im*log.(r))
end
(l::NPU)(x) = mult(l.Re, l.Im, l.g, x)




"""
    RealNPU(in::Int, out::Int; initRe=glorot_uniform, initg=init05)

NPU without imaginary weights.
"""
struct RealNPU{Tw<:AbstractMatrix,Tg<:AbstractVector}
    Re::Tw
    g::Tg
end

RealNPU(in::Int, out::Int; initRe=Flux.glorot_uniform, initg=init05) =
    RealNPU(initRe(out,in), initg(in))

Flux.@functor RealNPU

function mult(Re::AbstractMatrix{T}, g::AbstractVector{T}, x::AbstractArray{T}) where T
    g = min.(max.(g, 0), 1)

    r = abs.(x) .+ eps(T)
    r = g .* r .+ (1 .- g)

    k = max.(-sign.(x), 0) .* T(pi)
    k = g .* k

    z = exp.(Re * log.(r)) .* cos.(Re*k)
end

(l::RealNPU)(x) = mult(l.Re, l.g, x)




"""
    NaiveNPU(in::Int, out::Int; initRe=glorot_uniform, initIm=zeros)

`NPU` without relevance gating mechanism.
"""
struct NaiveNPU{T<:AbstractMatrix}
    Re::T
    Im::T
end

function NaiveNPU(in::Int, out::Int; initRe=glorot_uniform, initIm=Flux.zeros)
    Re = initRe(out, in)
    Im = initIm(out, in)
    NaiveNPU(Re, Im)
end

Flux.@functor NaiveNPU

function mult(Re::AbstractMatrix{T}, Im::AbstractMatrix{T}, x::AbstractArray{T}) where T
    r = abs.(x) .+ eps(T)
    k = signclip(x)
    exp.(Re*log.(r) - Im*k) .* cos.(Re*k + Im*log.(r))
end

(l::NaiveNPU)(x) = mult(l.Re, l.Im, x)

function ChainRulesCore.rrule(::typeof(mult), Re::AbstractMatrix{T}, Im::AbstractMatrix{T}, x::AbstractArray{T}) where T
    r  = abs.(x) .+ eps(T)
    k  = signclip(x)
    ex = Re*log.(r) - Im*k
    cx = Re*k + Im*log.(r)
    z  = exp.(ex) .* cos.(cx)

    function mult_pullback(ΔΩ::AbstractVector)
        sx  = 1 ./ sign.(x)
        a   = exp.(ex) .* sin.(cx)
        dX  = sx' .* (Re .* z - Im .* a)
        dRe = z * log.(r)' - a * k'
        dIm = z * k' - a * log.(r)'
        (NO_FIELDS,
         @thunk(dRe .* reshape(ΔΩ,:,1)),
         @thunk(dIm .* reshape(ΔΩ,:,1)),
         @thunk(dX' * ΔΩ))
    end

    z, mult_pullback
end



"""
  RealNaiveNPU(in::Int, out::Int; initRe=glorot_uniform)

NaiveNPU without imaginary weights.
"""
struct RealNaiveNPU{T<:AbstractMatrix}
    Re::T
end

RealNaiveNPU(in::Int, out::Int; initRe=glorot_uniform) = RealNaiveNPU(initRe(out,in))
Flux.@functor RealNaiveNPU

function mult(Re::AbstractMatrix{T}, x::AbstractArray{T}) where T
    r = abs.(x) .+ eps(T)
    k = max.(-sign.(x), 0) .* T(pi)
    exp.(Re * log.(r)) .* cos.(Re*k)
end

(l::RealNaiveNPU)(x) = mult(l.Re, x)



init05(s...) = Flux.ones(s...) ./ 2


# TODO: add in other layers and write rrule
signclip(x::AbstractArray{T}) where T = max.(-sign.(x), 0) * T(π)
gateclip(g::AbstractVector) = min.(max.(g, 0), 1)

#function ChainRulesCore.rrule(::typeof(gateclip), g)
#    ghat = gateclip(g)
#    gateclip_pullback(ΔΩ) = (NO_FIELDS, One() .* ΔΩ)
#    #gateclip_pullback(ΔΩ) = (NO_FIELDS, fill!(similar(g),1))
#    return ghat, gateclip_pullback
#end


function _repr(l)
    T = string(nameof(typeof(l)))
    "$T(in=$(size(l.Re,2)), out=$(size(l.Re,1)))"
end
Base.show(io::IO, l::NPU) = print(io, _repr(l))
Base.show(io::IO, l::NaiveNPU) = print(io, _repr(l))
Base.show(io::IO, l::RealNPU) = print(io, _repr(l))
Base.show(io::IO, l::RealNaiveNPU) = print(io, _repr(l))
