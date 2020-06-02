export NMU

"""
    NMU(in::Int, out::Int; init=rand)

Neural multiplication unit. Can represent multiplications between inputs.
Weights are clipped to [0,1].

As introduced in in https://openreview.net/pdf?id=H1gNOeHKPS
"""
struct NMU
    W::AbstractMatrix
end

NMU(in::Int, out::Int; init=rand) = NMU(init(out,in))


# softmaximum(x,y;k=10) = log(exp(k*x) + exp(k*y)) / k
# softminimum(x,y;k=10) = -softmaximum(-x,-y)
# weights(m::NMU) = softminimum.(softmaximum.(m.W, 0), 1)

weights(m::NMU) = min.(max.(m.W, 0), 1)

function (m::NMU)(x::AbstractVector)
    W = weights(m)
    z = W .* reshape(x,1,:) .+ 1 .- W
    dropdims(prod(z, dims=2), dims=2)
end

function (m::NMU)(x::AbstractMatrix)
    buf = Zygote.Buffer(x, size(m.W,1), size(x,2))
    for i in 1:size(x,2)
        buf[:,i] = m(x[:,i])
    end
    copy(buf)
end

Flux.@functor NMU
