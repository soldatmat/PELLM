using FileIO
using PyCall
using DESilico

pickle = pyimport("pickle")

include("../de/utils.jl")
include("best_so_far.jl")

folder_path = joinpath(@__DIR__, "..", "de", "data", "perceptron", "PhoQ", "sample")

fitness_progressions = Vector{Vector{Float64}}([])
best_so_far_progressions = Vector{Vector{Float64}}([])

indexes = 1:200
data = load(joinpath(folder_path, "history_$(indexes[1]).jld2"))
top_fitness = data["top_fitness"]
prediction_history = data["prediction_history"]
best_so_far, budgets_control = get_best_so_far(top_fitness, prediction_history)
for i in indexes
    data = load(joinpath(folder_path, "history_$i.jld2"))
    top_fitness = data["top_fitness"]
    prediction_history = data["prediction_history"]

    best_so_far, budgets = get_best_so_far(top_fitness, prediction_history)
    budgets == budgets_control || println("Inconsistent budgets!")

    push!(fitness_progressions, top_fitness)
    push!(best_so_far_progressions, best_so_far)
end
fitness_progressions = reduce(vcat, transpose.(fitness_progressions))
best_so_far_progressions = reduce(vcat, transpose.(best_so_far_progressions))

save(
    joinpath(folder_path, "results_200.jld2"),
    "fitness_progressions", fitness_progressions,
    "best_so_far_progressions", best_so_far_progressions,
    "budgets", budgets_control
)

@pywith pybuiltin("open")(joinpath(folder_path, "results_200.pkl"), "wb") as f begin
    pickle.dump([
            fitness_progressions,
            best_so_far_progressions,
            budgets_control
        ], f)
end
