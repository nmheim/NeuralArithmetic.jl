export NAC

"""
    NAC(in::Int, out::Int; initW=glorot_uniform, initM=glorot_uniform)

Neural Accumulator as proposed in https://arxiv.org/abs/1808.00508
"""
struct NAC{T<:AbstractMatrix}
    W::T
    M::T
end

function NAC(in::Int, out::Int;
             initW=glorot_uniform, initM=glorot_uniform)
    W = initW(out, in)
    M = initM(out, in)
    NAC(W, M)
end

weights(nac::NAC) = tanh.(nac.W) .* σ.(nac.M)

function (nac::NAC)(x)
    _W = weights(nac)
    _W*x
end

function Base.show(io::IO, l::NAC)
    in = size(l.W, 2)
    out = size(l.W, 1)
    print(io, "NAC(in=$in, out=$out)")
end

Flux.@functor NAC
