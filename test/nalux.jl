@testset "NALUX" begin
    nalu = NALUX(2, 3, Dense(2, 3))
    batch = 30
    generate() = randn(Float32, 2, batch) .* 4
    f(x) = x[1,:] .* x[2,:]
    x = generate()
    z = nalu(x)

    @test size(z) == (3,batch)
    
    ps = params(nalu)
    @test length(ps) == 5

    # TODO: fix NALUX tests
    # mse(x) = Flux.mse(nalu(x), reshape(repeat(f(x), inner=(3,1)),3,batch))
    # reg(;β=0.01f0) = β*sum(norm, [nalu.A,nalu.rM,nalu.iM])
    # function bce(x;β=0.001f0)
    #     g = gate(nalu,x)
    #     β*sum(Flux.binarycrossentropy.(g,g))
    # end

    # loss(x) = mse(x) + reg() + bce(x)
    # #loss(x) = mse(x)
    # opt = ADAM(0.01)
    # train_data = [(generate(),) for _ in 1:10000]
    # Flux.train!(loss, ps, train_data, opt, cb=()->(@info loss(x)))

    # display(nalu.rM)
    # display(nalu.iM)
    # display(nalu.A)
    # display(gate(nalu,x))
    # @test all(isapprox.(nalu.rM, [1.0 1.0; 1.0 1.0; 1.0 1.0], atol=1e-1))
    # @test all(isapprox.(gate(nalu, x[:,1]), [0.0, 0.0, 0.0], atol=1e-1))
end
