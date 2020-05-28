export FastGatedNPUX, FastGatedNPU


struct FastGatedNPUX{F} <: DiffEqFlux.FastLayer
    in::Int
    out::Int
    initial_params::F
    function FastGatedNPUX(in::Int, out::Int;
                           initRe=Flux.glorot_uniform, initIm=Flux.glorot_uniform)
        initial_params() = vcat(vec(initRe(out,in)),
                                vec(initIm(out,in)),
                                Flux.ones(in)/2)
        new{typeof(initial_params)}(in, out, initial_params)
    end
end

DiffEqFlux.paramlength(f::FastGatedNPUX) = (f.out*f.in)*2 + f.in
DiffEqFlux.initial_params(f::FastGatedNPUX) = f.initial_params()

function _restructure(f::FastGatedNPUX, p::AbstractVector)
    len = f.out * f.in
    Re = reshape(p[1:len], f.out, f.in)
    Im = reshape(p[(len+1):(2*len)], f.out, f.in)
    g  = p[(2*len+1):end]
    return (Re,Im,g)
end

function (f::FastGatedNPUX)(x::AbstractVector,p::AbstractVector)
    (Re,Im,g) = _restructure(f,p)
    mult(Re,Im,g,x)
end

function (f::FastGatedNPUX)(x::AbstractMatrix,p::AbstractVector)
    (Re,Im,g) = _restructure(f,p)
    mult(Re,Im,g,x)
end



struct FastGatedNPU{F} <: DiffEqFlux.FastLayer
    in::Int
    out::Int
    initial_params::F
    function FastGatedNPU(in::Int, out::Int; init=Flux.glorot_uniform)
        initial_params() = vcat(vec(init(out,in)), Flux.ones(in)/2)
        new{typeof(initial_params)}(in, out, initial_params)
    end
end

DiffEqFlux.paramlength(f::FastGatedNPU) = (f.out*f.in) + f.in
DiffEqFlux.initial_params(f::FastGatedNPU) = f.initial_params()

function _restructure(f::FastGatedNPU, p::AbstractVector)
    len = f.out * f.in
    W = reshape(p[1:len], f.out, f.in)
    g = p[(len+1):end]
    return (W,g)
end

function (f::FastGatedNPU)(x::AbstractVector,p::AbstractVector)
    (W,g) = _restructure(f,p)
    mult(W,g,x)
end

function (f::FastGatedNPU)(x::AbstractMatrix,p::AbstractVector)
    (W,g) = _restructure(f,p)
    mult(W,g,x)
end
