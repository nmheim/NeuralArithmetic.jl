using Distributions
using LinearAlgebra
using NeuralArithmetic
using Flux
using Random

Random.seed!(1)

# define the learning task R²->R⁴
function task(x::Vector)
    a, b = x[1], x[2]
    [a+b, a*b, a/b, sqrt(b)]
end
task(x::Matrix) = mapslices(task, x, dims=1)

function generate_data(batchsize::Int)
    a = Uniform(-2,2)
    b = Uniform(0.1,2) # prevent zero/negative inputs to div/sqrt
    p = Product([a,b])
    x = Float32.(rand(p,batchsize))
    y = task(x)
    (x,y)
end

isize = 2     # input size
hsize = 6     # hidden size
osize = 4     # output size
β     = 1e-2  # L₁ regularization

batchsize = 100
nrsteps   = 10000
lr        = 1e-3

model = Chain(NPU(isize,hsize), NAU(hsize,osize))
ps    = Flux.params(model)
data  = [generate_data(batchsize) for _ in 1:nrsteps]
opt   = RMSProp(lr)
mse(x,y) = Flux.mse(model(x),y)
loss(x,y) = mse(x,y) + β*norm(ps,1)

cbs = Flux.throttle(()->(@info "training..." loss(data[1]...) mse(data[1]...)),1)
Flux.train!(loss, ps, data, opt, cb=cbs)

@info "NPU" model[1].Re model[1].Im model[1].g
@info "NAU" model[2].W

using Plots
p1 = heatmap(model[1].Re[end:-1:1,:], aspectratio=1, title="Learned NPU.Re", clim=(-1,1), c=:bluesreds)
p2 = heatmap(model[1].Im[end:-1:1,:], aspectratio=1, title="Learned NPU.Im", clim=(-1,1), c=:bluesreds)
p3 = heatmap(reshape(model[1].g,:,1), aspectratio=1, title="Learned NPU.g", clim=(-1,1), c=:bluesreds)
p4 = heatmap(model[2].W[end:-1:1,:], aspectratio=1, title="Learned NAU.W", clim=(-1,1), c=:bluesreds)

Re = [1 1; 1 -1; 0 0; 0 1; 1 0; 0 0]
Im = zeros(6,2)
g  = ones(2)
W  = [0 0 0 1 1 0;
      1 0 0 0 0 0;
      0 1 0 0 0 0;
      0 0 0 0.5 0 0]
p5 = heatmap(Re[end:-1:1,:], aspectratio=1, title="Perfect NPU.Re", clim=(-1,1), c=:bluesreds)
p6 = heatmap(Im[end:-1:1,:], aspectratio=1, title="Perfect NPU.Im", clim=(-1,1), c=:bluesreds)
p7 = heatmap(g[end:-1:1,:], aspectratio=1, title="Perfect NPU.g", clim=(-1,1), c=:bluesreds)
p8 = heatmap(W[end:-1:1,:], aspectratio=1, title="Perfect NAU.W", clim=(-1,1), c=:bluesreds)
plt1 = plot(p1,p2,p3,p4,p5,p6,p7,p8,layout=(2,4),size=(2000,1000))
