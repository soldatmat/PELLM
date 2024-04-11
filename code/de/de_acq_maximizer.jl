using BOSS

struct DEAcqMaximizer <: BOSS.AcqMaximizer end

function estimate_parameters(acq_maximizer::DEAcqMaximizer, acquisition::AcquisitionFunction, problem::BossProblem, options::BossOptions)
    acq = acquisition(problem, options)
    # acq: [3, 5, 12, 3] -> fitness
    problem.domain.bounds # ([1, 1, 1, 1], [20, 20, 20, 20])
    # TODO get fitness from each x
    # TODO return x, which has highest fitness
end
