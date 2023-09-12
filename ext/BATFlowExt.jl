# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATFlowExt

@static if isdefined(Base, :get_extension)
    using NestedSamplers
else
    using ..NestedSamplers
end

using BAT
using HeterogeneousComputing

BAT.pkgext(::Val{:NestedSamplers}) = BAT.PackageExtension{:NestedSamplers}()

using BAT: AnyMeasureOrDensity, AbstractMeasureOrDensity
using BAT: ENSBound, ENSNoBounds, ENSEllipsoidBound, ENSMultiEllipsoidBound
using BAT: ENSProposal, ENSUniformly, ENSAutoProposal, ENSRandomWalk, ENSSlice 

using Statistics, StatsBase
using DensityInterface, InverseFunctions, ValueShapes
import Measurements

using Random


include("/ceph/groups/e4/users/wweber/public/.julia/dev/BAT/src/samplers/mcmc/NFproposaldist.jl")


end # module BATNestedSamplersExt
