using FileIO
using PyCall

pickle = pyimport("pickle")

include("best_so_far.jl")

folder_path = joinpath(@__DIR__, "..", "de", "data", "perceptron", "perceptron_distmax_finetune_phoq_01")
d = load(joinpath(folder_path, "history.jld2"))
top_fitness = d["top_fitness"]
prediction_history = d["prediction_history"]

best_so_far, budgets = get_best_so_far(top_fitness, prediction_history)

save(
    joinpath(folder_path, "best_so_far.jld2"),
    "best_so_far", best_so_far,
    "budgets", budgets
)

@pywith pybuiltin("open")(joinpath(folder_path, "best_so_far.pkl"), "wb") as f begin
    pickle.dump([
            best_so_far,
            budgets
        ], f)
end

@pywith pybuiltin("open")(joinpath(folder_path, "fitness_progression.pkl"), "wb") as f begin
    pickle.dump([
            top_fitness
        ], f)
end
