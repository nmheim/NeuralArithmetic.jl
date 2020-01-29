export NALUX
export Bias
export CheckerboardGate

## HELPER LAYERS
struct CheckerboardGate{T,F}
    c::T
    b::T
    σ::F
end

CheckerboardGate(len::Int, σ=identity; init=zeros) =
    CheckerboardGate(init(Float32,len), init(Float32,len), σ)

function (m::CheckerboardGate)(x::AbstractArray)
    c, b, σ = m.c, m.b, m.σ
    s = sign.(x)
    s = prod(s, dims=1)
    gc = σ.(c)
    gb = σ.(b)

    checkerboard = gc .* s .- (1 .- gc) .* s
    gb .+ (1 .- gb) .* checkerboard
end
Flux.@functor CheckerboardGate


struct Bias{T,F}
    b::T
    σ::F
end

Bias(len::Int, σ=identity; init=zeros) = Bias(init(Float32,len), σ)
(m::Bias)() = m.σ.(m.b)
(m::Bias)(x::AbstractArray) = m()
Flux.@functor Bias


"""
    NALUX(in::Int, out::Int, S::T; initA=glorot_uniform, initM=glorot_uniform) where T

Extended Neural Arithmetic Logic Unit. Experimental. Can recover signs
(currently only for two inputs).
"""
struct NALUX{T}
    A::AbstractMatrix
    M::AbstractMatrix
    S::T
    ϵ::Real
end

NALUX(A::AbstractMatrix, M::AbstractMatrix, S::T) where T =
    NALUX{T}(A,M,S,Float32(1e-8))

function NALUX(in::Int, out::Int, S::T;
                   initA=glorot_uniform,
                   initM=glorot_uniform) where T
    A = initA(out, in)
    M = initM(out, in)
    NALUX(A, M, S)
end

function NALUX(in::Int, out::Int)
    n = max(in,out)
    A = Array(Diagonal(ones(Float32, n, n)))
    A = A[1:out,1:in]
    M = Array(Diagonal(ones(Float32, n, n)))
    M = M[1:out,1:in]
    S = CheckerboardGate(out, σ)
    NALUX(A, M, S)
end

addition(nalu::NALUX, x) = nalu.A * x
multiplication(nalu::NALUX, x) = exp.(nalu.M * log.(abs.(x) .+ nalu.ϵ))
sign(nalu::NALUX, x) = nalu.S(x)

function (nalu::NALUX)(x::AbstractArray)
    a = addition(nalu, x)
    m = multiplication(nalu, x)
    s = sign(nalu, x)
    a .+  m .* s
end

function Base.show(io::IO, l::NALUX)
    in = size(l.A, 2)
    out = size(l.A, 1)
    print(io, "NALUX(in=$in, out=$out)")
end

Flux.@functor NALUX
