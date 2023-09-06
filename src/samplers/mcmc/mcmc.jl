# This file is a part of BAT.jl, licensed under the MIT License (MIT).

include("mcmc_weighting.jl")
include("proposaldist.jl")
if isfile("/net/e4-nfs-home.e4.physik.tu-dortmund.de/home/wweber/Documents/test.jls")
    println("Use existing Flow for sampling")
    include("NFproposaldist.jl")
end
include("mcmc_sampleid.jl")
include("mcmc_algorithm.jl")
include("mcmc_noop_tuner.jl")
include("mcmc_stats.jl")
include("mcmc_convergence.jl")
include("chain_pool_init.jl")
include("multi_cycle_burnin.jl")
include("mcmc_sample.jl")
include("mh/mh.jl")
