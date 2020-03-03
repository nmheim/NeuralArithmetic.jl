export NMUX

"""
    NMUX(in::Int, out::Int; initRe=glorot_uniform, initIm=zeros)

NMU that can learn any power function by using a complex multiplication matrix.
"""

struct NMUX
    Re::AbstractMatrix
    Im::AbstractMatrix
end

function NMUX(in::Int, out::Int, initRe=glorot_uniform, initIm=Flux.zeros)
    Re = initRe(out, in)
    Im = initIm(out, in)
    NMUX(Re, Im)
end

Flux.@functor NMUX

function mult(Re::AbstractMatrix{T}, Im::AbstractMatrix{T}, x::AbstractArray{T}) where T
    r = abs.(x)
    k = map(i -> T(i < 0 ? pi : 0.0), x)
    exp.(Re*log.(r) - Im*k) .* cos.(Re*k + Im*log.(r))
end

(l::NMUX)(x) = mult(l.Re, l.Im, x)
