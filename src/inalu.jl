export iNALU


"""
    iNALU(in::Int, out::Int; initNAC=glorot_uniform, initG=glorot_uniform, ϵ=1e-7, ω=20)

Improved NALU that can process negative numbers by recovering the multiplication
sign. Implemented as suggested in: https://arxiv.org/abs/2003.07629
"""
struct iNALU
    a_nac::NAC
    m_nac::NAC
    G::AbstractMatrix
    ϵ::Real
    ω::Real
end

Flux.@functor iNALU

iNALU(a_nac::NAC, m_nac::NAC, G::AbstractMatrix) = iNALU(a_nac, m_nac, G, 1e-7, 20)

function iNALU(in::Int, out::Int;
              initNAC=glorot_uniform, initG=glorot_uniform, ϵ=1e-7, ω=20)
    a_nac = NAC(in, out, initW=initNAC, initM=initNAC)
    m_nac = NAC(in, out, initW=initNAC, initM=initNAC)
    G = initG(out, in)
    iNALU(a_nac,m_nac,G,ϵ,ω)
end

function (m::iNALU)(x)
    # add / mult paths
    a = m.a_nac(x)
    m = exp.(min.(m.m_nac(log.(max.(abs.(x),m.ϵ))),m.ω))

    # sign recovery
    W  = abs.(weights(m.m_nac))
    sm = sign.(x) .* W .+ 1 .- W
    s  = vec(prod(sm, dims=2))

    # add/mult gate
    g = σ.(m.G*x .+ m.b)

    g .* a .+ (1.0 .- g) .* m .* s
end

function Base.show(io::IO, l::iNALU)
    in = size(l.G, 2)
    out = size(l.G, 1)
    print(io, "iNALU(in=$in, out=$out)")
end
