using Distributions
using LinearAlgebra
using NeuralArithmetic
using Flux
using Random

Random.seed!(0)

# define the learning task R²->R⁴
function task(x::Vector)
    a, b = x[1], x[2]
    [a+b, a*b, a/b, sqrt(b)]
end
task(x::Matrix) = mapslices(task, x, dims=1)

function generate_data(batchsize::Int)
    a = Uniform(-2,2)
    b = Uniform(0.01,2) # prevent zero/negative inputs to div/sqrt
    p = Product([a,b])
    x = Float32.(rand(p,batchsize))
    y = task(x)
    (x,y)
end

isize = 2     # input size
hsize = 6     # hidden size
osize = 4     # output size
β     = 1e-4  # L₁ regularization

batchsize = 100
nrsteps   = 50000
lr        = 1e-3

model = Chain(NPU(isize,hsize), NAU(hsize,osize))
ps    = Flux.params(model)
data  = [generate_data(batchsize) for _ in 1:nrsteps]
opt   = ADAM(lr)
mse(x,y) = Flux.mse(model(x),y)
loss(x,y) = mse(x,y) + β*norm(ps,1)

cb = Flux.throttle(()->(@info "training..." loss(data[1]...) mse(data[1]...)),1)
Flux.train!(loss, ps, data, opt, cb=cb)

@info "NPU" model[1].Re model[1].Im model[1].g
@info "NAU" model[2].W

using Plots
pyplot()
p1 = heatmap(model[1].Re, yflip=true, aspectratio=1,
             ylabel="NPU Re", clim=(-1,1), c=:bluesreds, colorbar=false)
p2 = heatmap(model[1].Im, yflip=true, aspectratio=1, title="Learned Solution",
             ylabel="NPU Im", clim=(-1,1), c=:bluesreds, colorbar=false)
p4 = heatmap(model[2].W, yflip=true, aspectratio=1,
             ylabel="NAU W", clim=(-1,1), c=:bluesreds)

Re = [1 -1; 1 1; 0 0.5; 1 0; 0 0; 0 1]
Im = zeros(6,2)
g  = ones(2)
W  = [0 0 0 1 0 1;
      0 1 0 0 0 0;
      1 0 0 0 0 0;
      0 0 1 0 0 0]
p5 = heatmap(Re, yflip=true, aspectratio=1,
             ylabel="NPU Re", clim=(-1,1), c=:bluesreds, colorbar=false)
p6 = heatmap(Im, yflip=true, aspectratio=1, title="Perfect Solution",
             ylabel="NPU Im", clim=(-1,1), c=:bluesreds, colorbar=false)
p8 = heatmap(W, yflip=true, aspectratio=1,
             ylabel="NAU W", clim=(-1,1), c=:bluesreds)
plt1 = plot(p1,p2,p4,p5,p6,p8,layout=(2,3), size=(800,370))
