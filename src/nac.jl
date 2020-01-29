export NAC

"""
    NAC(in::Int, out::Int; initW=glorot_uniform, initM=glorot_uniform)

Neural Accumulator. Special case of affine layer in which the parameters
are encouraged to be close to {-1, 0, 1}.

Paper: https://arxiv.org/abs/1808.00508
"""
struct NAC
    W::AbstractMatrix
    M::AbstractMatrix
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
