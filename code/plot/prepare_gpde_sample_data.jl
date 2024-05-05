using FileIO
using DESilico
using BOSS
using PyCall

pickle = pyimport("pickle")

include("../de/utils.jl")

data_path = joinpath(@__DIR__, "..", "de", "data", "gpde", "PhoQ", "sample")

fitness_progressions = Vector{Vector{Float64}}([])
for i = 1:200
    problem = load(joinpath(data_path, "gp_$i.jld2"))["problem"]
    history = reconstruct_history(vec(problem.data.Y))
    push!(fitness_progressions, history)
end
fitness_progressions = reduce(vcat, transpose.(fitness_progressions))

save_path = joinpath(@__DIR__, "..", "de", "data", "gpde", "PhoQ", "sample", "results_200.pkl")
@pywith pybuiltin("open")(save_path, "wb") as f begin
    pickle.dump([
            fitness_progressions
        ], f)
end
