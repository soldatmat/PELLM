using FileIO
using BOSS
using PyCall

pickle = pyimport("pickle")
include("../de/boes/lib/boss_utils.jl")

# Define data path
dataset = "TrpB" # GB1, PhoQ, TrpB
change = "ucb_onehot_hyper" # ucb, onehot, ucb_onehot, ucb_onehot_hyper

global n_missing = 0
global n_incomplete = 0


data_path = joinpath(@__DIR__, "..", "..", "..", "BOES", "benchmarks", "ablation", change, dataset)
save_name = change * ".pkl"

fitness_progressions = Vector{Vector{Float64}}([])
for i = 1:100
    filename = joinpath(data_path, "variant_$i.jld2")
    if !isfile(filename)
        global n_missing += 1
        continue
    end
    problem = load(filename)["problem"]

    history = get_gp_fitness_progression(problem.data.Y)
    if length(history) < 125
        println(filename)
        println(length(history))
        global n_incomplete += 1
        continue
    end
    push!(fitness_progressions, history[1:125])
end
if n_missing > 0
    println("Some files are missing. Total missing files: $n_missing")
end
if n_incomplete > 0
    println("Some files are missing. Total missing files: $n_incomplete")
end
fitness_progressions = reduce(vcat, transpose.(fitness_progressions))

# fitness_progressions = fitness_progressions[:, 1:200]

save_path = joinpath("..", "..", "..", "evaluation", "boes", dataset, save_name)
@pywith pybuiltin("open")(save_path, "wb") as f begin
    pickle.dump([
            fitness_progressions
        ], f)
end

display(n_missing)
display(n_incomplete)
