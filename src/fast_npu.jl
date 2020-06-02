export FastNPU, FastRealNPU

"""
  FastNPU(in::Int, out::Int; initRe=glorot_uniform, initIm=Flux.zeros) <: FastLayer

Neural Power Unit that can learn arbitrary power functions by using a complex
weights. Uses gating on inputs to simplify learning. In 1D the layer looks
like:

    g = min(max(g, 0), 1)
    r = abs(x) + eps(T)
    r = g*r + (1-g)*T(1)
    k = r < 0 ? pi : 0.0
    exp(Re*log(r) - Im*k) * cos(Re*k + Im*log(r))

FastLayer meant for use with DiffEqFlux.
"""
struct FastNPU{F} <: DiffEqFlux.FastLayer
    in::Int
    out::Int
    initial_params::F
    function FastNPU(in::Int, out::Int;
                           initRe=Flux.glorot_uniform, initIm=Flux.glorot_uniform)
        initial_params() = vcat(vec(initRe(out,in)),
                                vec(initIm(out,in)),
                                Flux.ones(in)/2)
        new{typeof(initial_params)}(in, out, initial_params)
    end
end

DiffEqFlux.paramlength(f::FastNPU) = (f.out*f.in)*2 + f.in
DiffEqFlux.initial_params(f::FastNPU) = f.initial_params()

function _restructure(f::FastNPU, p::AbstractVector)
    len = f.out * f.in
    Re = reshape(p[1:len], f.out, f.in)
    Im = reshape(p[(len+1):(2*len)], f.out, f.in)
    g  = p[(2*len+1):end]
    return (Re,Im,g)
end

function (f::FastNPU)(x::AbstractVector,p::AbstractVector)
    (Re,Im,g) = _restructure(f,p)
    mult(Re,Im,g,x)
end

function (f::FastNPU)(x::AbstractMatrix,p::AbstractVector)
    (Re,Im,g) = _restructure(f,p)
    mult(Re,Im,g,x)
end


"""
    FastRealNPU(in::Int, out::Int; init=glorot_uniform) <: FastLayer

NPU without imaginary weights. FastLayer meant for use with DiffEqFlux.
"""
struct FastRealNPU{F} <: DiffEqFlux.FastLayer
    in::Int
    out::Int
    initial_params::F
    function FastRealNPU(in::Int, out::Int; init=Flux.glorot_uniform)
        initial_params() = vcat(vec(init(out,in)), Flux.ones(in)/2)
        new{typeof(initial_params)}(in, out, initial_params)
    end
end

DiffEqFlux.paramlength(f::FastRealNPU) = (f.out*f.in) + f.in
DiffEqFlux.initial_params(f::FastRealNPU) = f.initial_params()

function _restructure(f::FastRealNPU, p::AbstractVector)
    len = f.out * f.in
    W = reshape(p[1:len], f.out, f.in)
    g = p[(len+1):end]
    return (W,g)
end

function (f::FastRealNPU)(x::AbstractVector,p::AbstractVector)
    (W,g) = _restructure(f,p)
    mult(W,g,x)
end

function (f::FastRealNPU)(x::AbstractMatrix,p::AbstractVector)
    (W,g) = _restructure(f,p)
    mult(W,g,x)
end
