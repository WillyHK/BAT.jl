using BAT, ValueShapes, Statistics, Clustering
using BAT: get_mcmc_tuning, nop_func, mcmc_init!, RNGPartition, _gen_chains, isvalidchain,
    tuning_init!, mcmc_iterate!, isviablechain, nsamples, tuning_postinit!

rng = bat_rng()
target = convert(AbstractDensity, BAT.MultimodalCauchy())
algorithm = MCMCSampling(trafo = NoDensityTransform())

density_notrafo = convert(AbstractDensity, target)
shaped_density, trafo = bat_transform(algorithm.trafo, density_notrafo)
density = unshaped(shaped_density)
mcmc_algorithm = algorithm.mcalg


(chains, tuners, chain_outputs) = mcmc_init!(
    rng,
    mcmc_algorithm,
    density,
    algorithm.nchains,
    algorithm.init,
    get_mcmc_tuning(mcmc_algorithm),
    algorithm.nonzero_weights,
    algorithm.store_burnin ? algorithm.callback : nop_func
)



#=
nchains = algorithm.nchains
init_alg = algorithm.init
tuning_alg = get_mcmc_tuning(mcmc_algorithm)
nonzero_weights = algorithm.nonzero_weights
callback = algorithm.store_burnin ? algorithm.callback : nop_func
algorithm = mcmc_algorithm


@info "Trying to generate $nchains viable MCMC chain(s)."

initval_alg = InitFromTarget()

min_nviable = minimum(init_alg.init_tries_per_chain) * nchains
max_ncandidates = maximum(init_alg.init_tries_per_chain) * nchains

rngpart = RNGPartition(rng, Base.OneTo(max_ncandidates))

ncandidates = 0

dummy_initval = unshaped(bat_initval(rng, density, InitFromTarget()).result, varshape(density))
dummy_chain = MCMCIterator(deepcopy(rng), algorithm, density, 1, dummy_initval)
dummy_tuner = tuning_alg(dummy_chain)

chains = similar([dummy_chain], 0)
tuners = similar([dummy_tuner], 0)
outputs = similar([DensitySampleVector(dummy_chain)], 0)
cycle = 1

while length(tuners) < min_nviable && ncandidates < max_ncandidates
    n = min(min_nviable, max_ncandidates - ncandidates)
    @debug "Generating $n $(cycle > 1 ? "additional " : "")MCMC chain(s)."

    new_chains = _gen_chains(rngpart, ncandidates .+ (one(Int64):n), algorithm, density, initval_alg)

    filter!(isvalidchain, new_chains)

    new_tuners = tuning_alg.(new_chains)
    new_outputs = DensitySampleVector.(new_chains)
    tuning_init!.(new_tuners, new_chains)
    global ncandidates += n

    @debug "Testing $(length(new_tuners)) MCMC chain(s)."

    mcmc_iterate!(
        new_outputs, new_chains;
        max_nsteps = max(50, div(init_alg.nsteps_init, 5)),
        callback = callback,
        nonzero_weights = nonzero_weights
    )

    viable_idxs = findall(isviablechain.(new_chains))
    viable_tuners = new_tuners[viable_idxs]
    viable_chains = new_chains[viable_idxs]
    viable_outputs = new_outputs[viable_idxs]

    @debug "Found $(length(viable_idxs)) viable MCMC chain(s)."

    if !isempty(viable_tuners)
        mcmc_iterate!(
            viable_outputs, viable_chains;
            max_nsteps = init_alg.nsteps_init,
            callback = callback,
            nonzero_weights = nonzero_weights
        )

        nsamples_thresh = floor(Int, 0.8 * median([nsamples(chain) for chain in viable_chains]))
        good_idxs = findall(chain -> nsamples(chain) >= nsamples_thresh, viable_chains)
        @debug "Found $(length(viable_tuners)) MCMC chain(s) with at least $(nsamples_thresh) unique accepted samples."

        append!(chains, view(viable_chains, good_idxs))
        append!(tuners, view(viable_tuners, good_idxs))
        append!(outputs, view(viable_outputs, good_idxs))
    end

    global cycle += 1
end

length(tuners) < min_nviable && error("Failed to generate $min_nviable viable MCMC chains")

m = nchains
tidxs = LinearIndices(tuners)
n = length(tidxs)

modes = hcat(broadcast(samples -> Array(bat_findmode(samples, MaxDensitySampleSearch()).result), outputs)...)

final_chains = similar(chains, 0)
final_tuners = similar(tuners, 0)
final_outputs = similar(outputs, 0)

if 2 <= m < size(modes, 2)
    clusters = kmeans(modes, m, init = KmCentralityAlg())
    clusters.converged || error("k-means clustering of MCMC chains did not converge")

    mincosts = fill(Inf, m)
    chain_sel_idxs = fill(0, m)

    for i in tidxs
        j = clusters.assignments[i]
        if clusters.costs[i] < mincosts[j]
            mincosts[j] = clusters.costs[i]
            chain_sel_idxs[j] = i
        end
    end

    @assert all(j -> j in tidxs, chain_sel_idxs)

    for i in sort(chain_sel_idxs)
        push!(final_chains, chains[i])
        push!(final_tuners, tuners[i])
        push!(final_outputs, outputs[i])
    end
else
    @assert length(chains) == nchains
    resize!(final_chains, nchains)
    copyto!(final_chains, chains)

    @assert length(tuners) == nchains
    resize!(final_tuners, nchains)
    copyto!(final_tuners, outputs)

    @assert length(outputs) == nchains
    resize!(final_outputs, nchains)
    copyto!(final_outputs, outputs)
end

@info "Selected $(length(final_tuners)) MCMC chain(s)."
tuning_postinit!.(final_tuners, final_chains, final_outputs)

(chains = final_chains, tuners = final_tuners, outputs = final_outputs)
=#