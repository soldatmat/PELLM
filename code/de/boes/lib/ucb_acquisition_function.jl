using BOSS

# Implementation assumes one-dimensional output
struct UCB <: BOSS.AcquisitionFunction
    κ::Float64
end
# κ=2.576 corresponds to the 99% confidence bound under a Gaussian assumption
# (encouraging strong exploration).
function UCB(; κ=2.576)
    return UCB(κ)
end

function (ei::UCB)(problem::BossProblem, options::BossOptions)
    posterior = model_posterior(problem.model, problem.data)

    acq = function(x)
        μ, σ = posterior(x)
        # Implementation assumes one-dimensional output
        μ = μ[1]
        σ = σ[1]
        return μ + ei.κ * σ
    end

    return acq
end
