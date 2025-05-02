using FileIO
using BOSS
using PyCall

pickle = pyimport("pickle")
include("../de/boes/lib/boss_utils.jl")

# Define data path
dataset = "TrpB" # GB1, PhoQ, TrpB
method = "matern" # original, matern, conic, additive

# GB1: matern 91, conic=1:87, additive=1:92

folder_name = Dict{String, String}(
    "matern" => "M",
    "conic" => "C",
    "additive" => "A",
    "original" => "G",
)

global n_missing = [0, 0, 0, 0, 0, 0, 0]
#dims = ["1", "5", "10", "20", "50", "75", "100"]
dims = ["1028"] # save as 1280

for d in eachindex(dims)
dim = dims[d]

data_path = joinpath(@__DIR__, "..", "..", "..", "BOES", "benchmarks", "rerun", dataset, "original", folder_name[method], "dim_"*dim)
#save_name = method*"_"*dim*".pkl"
save_name = method * "_1280.pkl"

fitness_progressions = Vector{Vector{Float64}}([])
for i = 1:100
    filename = joinpath(data_path, "variant_$i.jld2")
    if !isfile(filename)
        global n_missing[d] += 1
        continue
    end
    problem = load(filename)["problem"]

    # !!! Fix data after BOES initialization bug !!!
    #Y_fixed = problem.data.Y
    #Y_fixed[1] = 0.

    history = get_gp_fitness_progression(problem.data.Y)
    if length(history) != 201
        global n_missing[d] += 1
        continue
    end
    push!(fitness_progressions, history)
end
if n_missing[d] > 0
    #error("Some files are missing. Total missing files: $n_missing[d]")
    println("Some files are missing. Total missing files: $(n_missing[d]). dim = $dim")
end
fitness_progressions = reduce(vcat, transpose.(fitness_progressions))

fitness_progressions = fitness_progressions[:, 1:200]

save_path = joinpath("..", "..", "..", "evaluation", "boes", dataset, save_name)
@pywith pybuiltin("open")(save_path, "wb") as f begin
    pickle.dump([
            fitness_progressions
        ], f)
end

end
display(dims)
display(n_missing)
