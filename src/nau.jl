export NAU

"""
    NAU(in::Int, out::Int; init=glorot_uniform)

Neural addition unit.

As suggested in https://openreview.net/pdf?id=H1gNOeHKPS
"""
struct NAU{T<:AbstractMatrix}
    W::T
end

NAU(in::Int, out::Int; init=glorot_uniform) = NAU(init(out,in))

(m::NAU)(x) = m.W * x

Flux.@functor NAU

Base.show(io::IO, l::NAU) = print(io,"NAU(in=$(size(l.W,2)), out=$(size(l.W,1)))")
