export NAU

"""
    NAU(in::Int, out::Int; init=glorot_uniform)

Neural addition unit.

Lacks the regularization suggested in https://openreview.net/pdf?id=H1gNOeHKPS
as it is intended to be used with ARD (automatic relevance determination)
"""
struct NAU
    W::AbstractMatrix
end

NAU(in::Int, out::Int; init=glorot_uniform) = NAU(init(out,in))

(m::NAU)(x) = m.W * x

Flux.@functor NAU
