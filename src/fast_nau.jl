export FastNAU

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

Base.show(io::IO, l::FastNAU) = print(io,"FastNAU(in=$(l.in), out=$(l.out))")
