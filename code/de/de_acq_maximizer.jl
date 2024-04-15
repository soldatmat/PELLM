using BOSS

struct DEAcqMaximizer <: BOSS.AcquisitionMaximizer
    variant_coords::Vector{Vector{Int}}
end

function BOSS.maximize_acquisition(acq_maximizer::DEAcqMaximizer, acquisition::BOSS.AcquisitionFunction, problem::BossProblem, options::BossOptions)
    acq = acquisition(problem, options) # acq: [3, 5, 12, 3] -> fitness
    #problem.domain.bounds # ([1, 1, 1, 1], [20, 20, 20, 20])
    variants = map(coords -> (coords, acq(coords)), acq_maximizer.variant_coords)
    sort!(variants, by=x -> x[2], rev=true)
    return variants[1][1]
    # for batch, retunr matrix, columns are data points
end
