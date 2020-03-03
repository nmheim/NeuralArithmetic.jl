export NALU

"""
    NALU(in::Int, out::Int; initNAC=glorot_uniform, initG=glorot_uniform, initb=glorot_uniform)

Neural Arithmetic Logic Unit. Layer that is capable of learing multiplication,
division, power functions, addition, and subtraction.

Paper: https://arxiv.org/abs/1808.00508
"""
struct NALU
    nac::NAC
    G::AbstractMatrix
    b::AbstractVector
    ϵ::Real
end

NALU(nac::NAC, G::AbstractMatrix, b::AbstractVector) = NALU(nac, G, b, 1e-8)

function NALU(in::Int, out::Int;
              initNAC=glorot_uniform, initG=glorot_uniform, initb=glorot_uniform)
    nac = NAC(in, out, initW=initNAC, initM=initNAC)
    G = initG(out, in)
    b = initb(out)
    NALU(nac, G, b)
end

add(nalu::NALU, x) = nalu.nac(x)
mult(nalu::NALU, x) = exp.(nalu.nac(log.(abs.(x) .+ nalu.ϵ)))
gate(nalu::NALU, x) = σ.(nalu.G*x .+ nalu.b)

function (nalu::NALU)(x)
    a = add(nalu, x)
    m = mult(nalu, x)
    g = gate(nalu, x)
    g .* a .+ (1.0 .- g) .* m
end

function Base.show(io::IO, l::NALU)
    in = size(l.G, 2)
    out = size(l.G, 1)
    print(io, "NALU(in=$in, out=$out)")
end

Flux.@functor NALU
