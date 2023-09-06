# This file is a part of BAT.jl, licensed under the MIT License (MIT).

#function proposaldist_logpdf end


#function proposal_rand! end

struct NFlowProposalDist{D<:Distribution{Multivariate},SamplerF,S<:Sampleable} <: AbstractProposalDist
    d::D
    sampler_f::SamplerF
    s::S

    function NFlowProposalDist{D,SamplerF}(d::D, sampler_f::SamplerF) where {D<:Distribution{Multivariate},SamplerF}
        s = sampler_f(d)
        new{D,SamplerF, typeof(s)}(d, sampler_f, s)
    end

end


NFlowProposalDist(d::D, sampler_f::SamplerF) where {D<:Distribution{Multivariate},SamplerF} =
    NFlowProposalDist{D,SamplerF}(d, sampler_f)

NFlowProposalDist(d::Distribution{Multivariate}) = NFlowProposalDist(d, bat_sampler)

NFlowProposalDist(D::Type{<:Distribution{Multivariate}}, varndof::Integer, args...) =
    NFlowProposalDist(D, Float64, varndof, args...)


Base.similar(q::NFlowProposalDist, d::Distribution{Multivariate}) =
    NFlowProposalDist(d, q.sampler_f)

function Base.convert(::Type{AbstractProposalDist}, q::NFlowProposalDist, T::Type{<:AbstractFloat}, varndof::Integer)
    varndof != totalndof(q) && throw(ArgumentError("q has wrong number of DOF"))
    q
end


get_cov(q::NFlowProposalDist) = get_cov(q.d)
set_cov(q::NFlowProposalDist, Σ::PosDefMatLike) = similar(q, set_cov(q.d, Σ))


function proposaldist_logpdf(
    pdist::NFlowProposalDist,
    v_proposed::AbstractVector,
    v_current::AbstractVector
)
    #params_diff = v_proposed .- v_current # TODO: Avoid memory allocation
    #logpdf(pdist.d, params_diff)
    return 1
end


global flow = nothing
function proposal_rand!(
    rng::AbstractRNG,
    pdist::NFlowProposalDist,
    v_proposed::Union{AbstractVector,VectorOfSimilarVectors},
    v_current::Union{AbstractVector,VectorOfSimilarVectors}
)
    if isnothing(flow)
        global flow = open(deserialize,"/net/e4-nfs-home.e4.physik.tu-dortmund.de/home/wweber/Documents/test.jls")
        println("x")
    end
    dim = length(v_proposed)
    rand!(rng, MvNormal(zeros(dim),I(dim)), flatview(v_proposed))
    v_proposed = flow(v_proposed)
    params_new_flat = flatview(v_proposed)
    #params_new_flat .+= flatview(v_current)
    #println(v_proposed)

    v_proposed
end

function issymmetric(pdist::NFlowProposalDist)
    true
end


ValueShapes.totalndof(pdist::NFlowProposalDist) = length(pdist.d)

LinearAlgebra.issymmetric(pdist::NFlowProposalDist) = issymmetric_around_origin(pdist.d)



#abstract type ProposalDistSpec end
#
#
#struct MvTDistProposal <: ProposalDistSpec
#    df::Float64
#end

#MvTDistProposal() = MvTDistProposal(1.0)


(ps::MvTDistProposal)(T::Type{<:AbstractFloat}, varndof::Integer) =
    NFlowProposalDist(MvTDist, T, varndof, convert(T, ps.df))

function NFlowProposalDist(::Type{MvTDist}, T::Type{<:AbstractFloat}, varndof::Integer, df = one(T))
    println("Sample with flow")
    Σ = PDMat(Matrix(ScalMat(varndof, one(T))))
    μ = Fill(zero(eltype(Σ)), varndof)
    M = typeof(Σ)
    d = Distributions.GenericMvTDist(convert(T, df), μ, Σ)
    NFlowProposalDist(d)
end


#struct UvTDistProposalSpec <: ProposalDistSpec
#    df::Float64
#end

(ps::UvTDistProposalSpec)(T::Type{<:AbstractFloat}, varndof::Integer) =
    GenericUvProposalDist(TDist(convert(T, ps.df)), fill(one(T), varndof))
