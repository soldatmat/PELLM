using BOSS

struct DEAcqMaximizer <: BOSS.AcquisitionMaximizer
    variant_coords::Vector{Vector{Int}}
    variant_index::Dict{Vector{Int},Int}

    function DEAcqMaximizer(variant_coords::Vector{Vector{Int}})
        variant_index = Dict(variant_coords .=> 1:length(variant_coords))
        new(variand_coords, variant_index)
    end
end

function BOSS.maximize_acquisition(acq_maximizer::DEAcqMaximizer, acquisition::BOSS.AcquisitionFunction, problem::BossProblem, options::BossOptions)
    acq = acquisition(problem, options) # acq: [3, 5, 12, 3] -> fitness
    variants = map(coords -> (coords, acq(coords)), acq_maximizer.variant_coords)
    map(x -> variants[acq_maximizer.variant_index[x]] = 0.0, eachcol(problem.data.X))
    sort!(variants, by=x -> x[2], rev=true)
    return variants[1][1]
    # for batch: return matrix where columns are data points
end
