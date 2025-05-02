function get_best_so_far(top_fitness::AbstractVector{Float64}, prediction_history::AbstractVector{Float64})
    batch_size = 24
    budgets = map(i -> i * batch_size, eachindex(prediction_history))
    best_so_far = map(i -> maximum([prediction_history[i], top_fitness[budgets[i]]]), eachindex(prediction_history))
    #budgets = budgets .+ 100
    return best_so_far, budgets
end
