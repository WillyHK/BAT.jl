# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test

Test.@testset "Package BAT" begin
    @info "XXXXXXXXXXXXXX test_utils"
    include("utils/test_utils.jl")
    @info "XXXXXXXXXXXXXX test_rngs"
    include("rngs/test_rngs.jl")
    @info "XXXXXXXXXXXXXX test_distributions"
    include("distributions/test_distributions.jl")
    @info "XXXXXXXXXXXXXX test_variates"
    include("variates/test_variates.jl")
    @info "XXXXXXXXXXXXXX test_transforms"
    include("transforms/test_transforms.jl")
    @info "XXXXXXXXXXXXXX test_densities"
    include("densities/test_densities.jl")
    @info "XXXXXXXXXXXXXX test_initvals"
    include("initvals/test_initvals.jl")
    @info "XXXXXXXXXXXXXX test_statistics"
    include("statistics/test_statistics.jl")
    @info "XXXXXXXXXXXXXX test_optimization"
    include("optimization/test_optimization.jl")
    @info "XXXXXXXXXXXXXX test_samplers"
    include("samplers/test_samplers.jl")
    @info "XXXXXXXXXXXXXX test_io"
    include("io/test_io.jl")
    @info "XXXXXXXXXXXXXX test_plotting"
    include("plotting/test_plotting.jl")
    @info "XXXXXXXXXXXXXX test_integration"
    include("integration/test_integration.jl")
end
