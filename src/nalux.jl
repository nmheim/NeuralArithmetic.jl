export NALUX

struct NALUX{T}
    A::AbstractArray
    M::AbstractArray{<:Complex}
    G::T
end

Flux.@functor NALUX

complex_glorot_uniform(s...) = Complex.(glorot_uniform(s...))
real_paramlength(l::NALUX) = length(l.A) + (length(l.M) * 2) + length(destructure(l.G)[1])
complex_paramlength(l::NALUX) = length(l.A) + length(l.M) + length(destructure(l.G)[1])
gate(l::NALUX, x) = σ.(l.G(x))

function (l::NALUX)(x::AbstractArray{<:Real})
    a = l.A * x
    m = real.(exp.(l.M * log.(Complex.(x))))
    g = gate(l, x)
    g .* a .+ (1 .- g) .* m
end

function NALUX(in::Int, out::Int, G;
              initA=glorot_uniform, initM=complex_glorot_uniform)
    A = initA(out, in)
    M = initM(out, in)
    l = NALUX(A, M, G)
end

function restructureM(ps, s)
    l = Int(length(ps)/2)
    M = reshape(ps[1:l], s) .+ reshape(ps[(l+1):end], s) .* im
end

function restructure(l::NALUX, ps::AbstractVector{<:Real})
    g_ps, restructureG = destructure(l.G)
    @assert length(g_ps) + length(l.A) + length(l.M)*2 == length(ps)
    A = reshape(ps[1:length(l.A)], size(l.A))
    M = restructureM(ps[(length(A)+1):(length(A)+length(l.M)*2)], size(l.M))
    G = restructureG(ps[(length(A)+length(M)*2+1):end])
    NALUX(A,M,G)
end

function restructure(l::NALUX, ps::AbstractVector{<:Complex})
    g_ps, restructureG = destructure(l.G)
    @assert length(g_ps) + length(l.A) + length(l.M) == length(ps)
    A = reshape(real.(ps[1:length(l.A)]), size(l.A))
    M = reshape(ps[(length(A)+1):(length(A)+length(l.M))], size(l.M))
    G = restructureG(real.(ps[(length(A)+length(M)+1):end]))
    NALUX(A,M,G)
end

(l::NALUX)(x, ps) = restructure(l, ps)(x)
