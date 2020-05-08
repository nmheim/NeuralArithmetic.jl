export NPUX, NPU, GatedNPU

"""
    NPU(in::Int, out::Int; initRe=glorot_uniform, initIm=zeros)

Neural Power Unit that can learn any power function by using a complex
multiplication matrix.
"""

struct NPUX
    Re::AbstractMatrix
    Im::AbstractMatrix
end

function NPUX(in::Int, out::Int; initRe=glorot_uniform, initIm=Flux.zeros)
    Re = initRe(out, in)
    Im = initIm(out, in)
    NPUX(Re, Im)
end

Flux.@functor NPUX

function mult(Re::AbstractMatrix{T}, Im::AbstractMatrix{T}, x::AbstractArray{T}) where T
    r = abs.(x) .+ eps(T)
    k = map(i -> T(i < 0 ? pi : 0.0), x)
    exp.(Re*log.(r) - Im*k) .* cos.(Re*k + Im*log.(r))
end

(l::NPUX)(x) = mult(l.Re, l.Im, x)


struct NPU
    W::AbstractMatrix
end

NPU(in::Int, out::Int; init=glorot_uniform) = NPU(init(out,in))
Flux.@functor NPU

function mult(W::AbstractMatrix{T}, x::AbstractArray{T}) where T
    r = abs.(x) .+ eps(T)
    k = map(i -> T(i < 0 ? pi : 0.0), x)
    exp.(W * log.(r)) .* cos.(W*k)
end

(l::NPU)(x) = mult(l.W, x)


"""
    GatedNPU(in::Int, out::Int; init=glorot_uniform)

Neural Power Unit that can learn any power function. Uses gating on inputs
to simplify learning. In 1D the layer looks like:

    g = min(max(g, 0), 1)
    r = abs(x) + eps(T)
    r = g*r + (1-g)*T(1)
    k = r < 0 ? pi : 0.0
    exp(W*log(r)) * cos(W*k)
"""
struct GatedNPU
    W::AbstractMatrix
    g::AbstractVector
end

GatedNPU(in::Int, out::Int; init=Flux.glorot_uniform) =
    GatedNPU(init(out,in), Flux.ones(in)/2)

Flux.@functor GatedNPU

function mult(W::AbstractMatrix{T}, g::AbstractVector{T}, x::AbstractArray{T}) where T
    g = min.(max.(g, 0), 1)
    r = abs.(x) .+ eps(T)
    r = g .* r .+ (1 .- g) .* T(1)
    k = map(i -> T(i < 0 ? pi : 0.0), x)
    z = exp.(W * log.(r)) .* cos.(W*k)
end
