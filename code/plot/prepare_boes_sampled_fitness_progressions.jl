using FileIO
using BOSS
using PyCall

pickle = pyimport("pickle")
include("../de/boes/lib/boss_utils.jl")

# Define data path
data_path = joinpath(@__DIR__, "..", "de", "data", "boes", "GB1", "sample")

fitness_progressions = Vector{Vector{Float64}}([])
for i = 1:200
    problem = load(joinpath(data_path, "boes_$i.jld2"))["problem"]
    history = get_gp_fitness_progression(problem.data.Y)
    push!(fitness_progressions, history)
end
fitness_progressions = reduce(vcat, transpose.(fitness_progressions))

save_path = save_path = joinpath(data_path, "fitness_progressions.pkl")
@pywith pybuiltin("open")(save_path, "wb") as f begin
    pickle.dump([
            fitness_progressions
        ], f)
end
