export NMU

"""
    NMU(in::Int, out::Int; init=rand)

Neural multiplication unit. Can represent multiplications between inputs.
Weights are clipped to [0,1].

As introduced in in https://openreview.net/pdf?id=H1gNOeHKPS
"""
struct NMU{T<:AbstractMatrix}
    W::T
end

NMU(in::Int, out::Int; init=rand) = NMU(init(out,in))

weights(m::NMU) = min.(max.(m.W, 0), 1)

function (m::NMU)(x::AbstractMatrix)
    W  = weights(m)
    Wr = reshape(W, size(W)..., 1)
    xr = reshape(x, 1, size(x)...)
    z  = Wr .* xr .+ 1 .- W
    dropdims(prod(z, dims=2), dims=2)
end

function (m::NMU)(x::AbstractVector)
    z = m(reshape(x,:,1))
    dropdims(z,dims=2)
end

Flux.@functor NMU

Base.show(io::IO, l::NMU) = print(io,"NMU(in=$(size(l.W,2)), out=$(size(l.W,1)))")
