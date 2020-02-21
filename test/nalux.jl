@testset "NALUX" begin
    nalu = NALUX(2, 3, Dense(2, 3))
    batch = 30
    generate() = randn(2, batch) .* 4
    f(x) = x[1,:] .* x[2,:]
    x = generate()
    z = nalu(x)

    @test size(z) == (3,batch)
    
    ps = params(nalu)
    @test length(ps) == 4

    mse(x) = Flux.mse(nalu(x), reshape(repeat(f(x), inner=(3,1)),3,batch))
    reg(;β=0.001f0) = β*sum(norm, [nalu.A,nalu.M])
    function bce(x;β=0.0001f0)
        g = gate(nalu,x)
        β*sum(Flux.binarycrossentropy.(g,g))
    end

    loss(x) = mse(x) + reg() + bce(x)
    opt = ADAM(0.01)
    train_data = [(generate(),) for _ in 1:20000]
    Flux.train!(loss, ps, train_data, opt)

    @test all(isapprox.(real.(nalu.M), [1.0 1.0; 1.0 1.0; 1.0 1.0], atol=1e-1))
    @test all(isapprox.(gate(nalu, x[:,1]), [0.0, 0.0, 0.0], atol=1e-1))

    len = NeuralArithmetic.real_paramlength(nalu)
    @test NeuralArithmetic.restructure(nalu, rand(Float32, len)) isa NALUX
    len = NeuralArithmetic.complex_paramlength(nalu)
    @test NeuralArithmetic.restructure(nalu, Complex.(rand(Float32, len))) isa NALUX
end
