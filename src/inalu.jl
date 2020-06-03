export iNALU


"""
    iNALU(in::Int, out::Int; initNAC=glorot_uniform, initG=glorot_uniform, ϵ=1e-7, ω=20)

Improved NALU that can process negative numbers by recovering the multiplication
sign. Implemented as suggested in: https://arxiv.org/abs/2003.07629
"""
struct iNALU{Tg<:AbstractMatrix,T<:Real}
    a_nac::NAC
    m_nac::NAC
    G::Tg
    ϵ::T
    ω::T
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

function Base.sign(nalu::iNALU, x::AbstractVector) 
    W  = abs.(weights(nalu.m_nac))
    sm = reshape(sign.(x),1,:) .* W .+ 1 .- W
    s  = vec(prod(sm, dims=2))
end

function Base.sign(nalu::iNALU, x::AbstractMatrix) 
    buf = Zygote.Buffer(x, size(nalu.G,1), size(x,2))
    for i in 1:size(x,2)
        buf[:,i] = sign(nalu, x[:,i])
    end
    copy(buf)
end

add(nalu::iNALU, x) = nalu.a_nac(x)
mult(nalu::iNALU, x) = exp.(min.(nalu.m_nac(log.(max.(abs.(x),nalu.ϵ))),nalu.ω))
gate(nalu::iNALU, x) = σ.(nalu.G*x)

function (nalu::iNALU)(x)
    a = add(nalu, x)
    m = mult(nalu, x)
    s = sign(nalu, x)
    g = gate(nalu, x)
    g .* a .+ (1.0 .- g) .* m .* s
end

Base.show(io::IO, l::iNALU) =
  print(io, "iNALU(in=$(size(l.G,2)), out=$(size(l.G,1)))")
