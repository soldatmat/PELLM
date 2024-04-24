using BOSS

function get_gp_fitness_progression(problem::BossProblem)
    length(problem.data.Y) == 0 && return Vector{Float64}([])
    history = problem.data.Y
    fitness_progression = Vector{Float64}(undef, length(history))
    fitness_progression[1] = problem.data.Y[1]
    map(i -> fitness_progression[i] = history[i] > fitness_progression[i-1] ? history[i] : fitness_progression[i-1], 2:length(history))
    return fitness_progression
end
