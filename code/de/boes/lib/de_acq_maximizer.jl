using BOSS

struct DEAcqMaximizer <: BOSS.AcquisitionMaximizer
    variant_coords::Vector{Vector{Int}}
    variant_index::Dict{Vector{Int},Int}
    parallel::Bool

    function DEAcqMaximizer(variant_coords::Vector{Vector{Int}}, variant_index::Dict{Vector{Int},Int}; parallel::Bool=false)
        new(variant_coords, variant_index, parallel)
    end

    function DEAcqMaximizer(variant_coords::Vector{Vector{Int}}; parallel::Bool=false)
        variant_index = Dict(variant_coords .=> 1:length(variant_coords))
        new(variant_coords, variant_index, parallel)
    end
end

function BOSS.maximize_acquisition(acq_maximizer::DEAcqMaximizer, acquisition::BOSS.AcquisitionFunction, problem::BossProblem, options::BossOptions)
    acq = acquisition(problem, options) # acq: x -> acquisition function value

    if acq_maximizer.parallel
        # Parallel acquisition function evaluation
        scores = Array{Float64,1}(undef, length(acq_maximizer.variant_coords))
        Threads.@threads for i in 1:length(acq_maximizer.variant_coords)
            scores[i] = acq(acq_maximizer.variant_coords[i])
        end
    else
        # Sequential acquisition function evaluation
        scores = map(coords -> acq(coords), acq_maximizer.variant_coords)
    end

    map(x -> scores[acq_maximizer.variant_index[x]] = 0.0, eachcol(problem.data.X))
    variants = map(i -> (acq_maximizer.variant_coords[i], scores[i]), eachindex(acq_maximizer.variant_coords))
    sort!(variants, by=x -> x[2], rev=true)
    return variants[1] # return tuple (x, acq(x)) where acq(x) is maximal
    # for batch: return matrix where columns are data points
end
