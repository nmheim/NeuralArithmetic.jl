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


struct FastNAU{F} <: DiffEqFlux.FastLayer
    in::Int
    out::Int
    initial_params::F
    function FastNAU(in::Int, out::Int; init=Flux.glorot_uniform)
        initial_params() = vec(init(out,in))
        new{typeof(initial_params)}(in, out, initial_params)
    end
end

DiffEqFlux.paramlength(f::FastNAU) = f.out * f.in
DiffEqFlux.initial_params(f::FastNAU) = f.initial_params()

(f::FastNAU)(x,p) = reshape(p, f.out, f.in) * x

Zygote.@adjoint function (f::FastNAU)(x,p)
    W = reshape(p, f.out, f.in)
    f(x,p), ȳ -> (nothing, W'ȳ, vec(ȳ * x'))
end
