export NALUX

"""
    NALUX(in::Int, out::Int, G; initA=glorot_uniform, initM=glorot_uniform)

Extends the NALU to work with negative and small numbers by using a complex
multiplication matrix.
"""
struct NALUX{T}
    A::AbstractArray
    rM::AbstractArray
    iM::AbstractArray
    G::T
end

function NALUX(in::Int, out::Int, G;
              initA=glorot_uniform, initM=glorot_uniform)
    A = initA(out, in)
    rM = initM(out, in)
    iM = zeros(eltype(rM), out, in)
    l = NALUX(A, rM, iM, G)
end

Flux.@functor NALUX

gate(l::NALUX, x) = σ.(l.G(x))

function mult(rM::AbstractMatrix{T}, iM::AbstractMatrix{T}, x::AbstractArray{T}) where T
    k = map(i -> T(i < 0 ? pi : 0.0), x)
    r = abs.(x)
    exp.(rM*log.(r) - iM*k) .* cos.(rM*k + iM*log.(r))
end

mult(l::NALUX, x) = mult(l.rM, l.iM, x)

function (l::NALUX)(x::AbstractArray{<:Real})
    a = l.A * x
    m = mult(l, x)
    g = gate(l, x)
    g .* a .+ (1 .- g) .* m
end
