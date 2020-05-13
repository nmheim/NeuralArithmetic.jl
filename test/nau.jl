@testset "NAU" begin
    nau = NAU(2,3) |> gpu
    x = rand(2,10) |> gpu
    z = nau(x)
    @test size(z) == (3,10)

    ps = params(nau)
    @test length(ps) == 1

    loss() = sum(abs2, nau(x) .- reshape(x[1,:],1,:))
    opt = ADAM()
    train_data = Iterators.repeated((), 10000)
    Flux.train!(loss, ps, train_data, opt)

    nau = nau |> cpu
    @test all(isapprox.(nau.W, [1.0 0.0; 1.0 0.0; 1.0 0.0], atol=1e-4))
end
