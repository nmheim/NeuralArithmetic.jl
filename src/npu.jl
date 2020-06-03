export NaiveNPU, RealNaiveNPU, RealNPU, NPU

"""
  NPU(in::Int, out::Int; initRe=glorot_uniform, initIm=Flux.zeros)

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
                   initRe=glorot_uniform, initIm=Flux.zeros)
   Re = initRe(out, in) 
   Im = initIm(out, in)
   g  = Flux.ones(in)/2
   NPU(Re,Im,g)
end

Flux.@functor NPU

function mult(Re::AbstractMatrix{T}, Im::AbstractMatrix{T}, g::AbstractVector{T}, x::AbstractArray{T}) where T
    g = min.(max.(g, 0), 1)

    r = abs.(x) .+ eps(T)
    r = g .* r .+ (1 .- g)

    k = max.(-sign.(x), 0) .* T(pi)
    k = g .* k

    exp.(Re*log.(r) - Im*k) .* cos.(Re*k + Im*log.(r))
end
(l::NPU)(x) = mult(l.Re, l.Im, l.g, x)


"""
    RealNPU(in::Int, out::Int; init=glorot_uniform)

NPU without imaginary weights.
"""
struct RealNPU{Tw<:AbstractMatrix,Tg<:AbstractVector}
    W::Tw
    g::Tg
end

RealNPU(in::Int, out::Int; init=Flux.glorot_uniform) =
    RealNPU(init(out,in), Flux.ones(in)/2)

Flux.@functor RealNPU

function mult(W::AbstractMatrix{T}, g::AbstractVector{T}, x::AbstractArray{T}) where T
    g = min.(max.(g, 0), 1)

    r = abs.(x) .+ eps(T)
    r = g .* r .+ (1 .- g)

    k = max.(-sign.(x), 0) .* T(pi)
    k = g .* k

    z = exp.(W * log.(r)) .* cos.(W*k)
end

(l::RealNPU)(x) = mult(l.W, l.g, x)


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
    k = max.(-sign.(x), 0) .* T(pi)
    exp.(Re*log.(r) - Im*k) .* cos.(Re*k + Im*log.(r))
end

(l::NaiveNPU)(x) = mult(l.Re, l.Im, x)


"""
  RealNaiveNPU(in::Int, out::Int; init=glorot_uniform)

NaiveNPU without imaginary weights.
"""
struct RealNaiveNPU{T<:AbstractMatrix}
    W::T
end

RealNaiveNPU(in::Int, out::Int; init=glorot_uniform) = RealNaiveNPU(init(out,in))
Flux.@functor RealNaiveNPU

function mult(W::AbstractMatrix{T}, x::AbstractArray{T}) where T
    r = abs.(x) .+ eps(T)
    k = max.(-sign.(x), 0) .* T(pi)
    exp.(W * log.(r)) .* cos.(W*k)
end

(l::RealNaiveNPU)(x) = mult(l.W, x)



Base.show(io::IO, l::NPU) = print(io,"NPU(in=$(size(l.Re,2)), out=$(size(l.Re,1))")
Base.show(io::IO, l::NaiveNPU) = print(io,"NaiveNPU(in=$(size(l.Re,2)), out=$(size(l.Re,1))")
Base.show(io::IO, l::RealNPU) = print(io,"RealNPU(in=$(size(l.W,2)), out=$(size(l.W,1))")
Base.show(io::IO, l::RealNaiveNPU) = print(io,"RealNaiveNPU(in=$(size(l.W,2)), out=$(size(l.W,1))")
