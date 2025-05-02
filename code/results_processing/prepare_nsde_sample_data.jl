using FileIO
using PyCall
using DESilico

pickle = pyimport("pickle")

include("../de/utils.jl")

dataset = "PhoQ" # GB1, PhoQ
folder_path = joinpath(@__DIR__, "..", "de", "data", "neighborhood_de", dataset, "sample")

data = load(joinpath(folder_path, "results_sample_1000.jld2"))
history = data["history"]
top_variant = reconstruct_history.(history)
top_fitness = map.(variant -> variant.fitness, top_variant)

save(
    joinpath(folder_path, "fitness_progressions.jld2"),
    "fitness_progressions", top_fitness,
)

@pywith pybuiltin("open")(joinpath(folder_path, "fitness_progressions.pkl"), "wb") as f begin
    pickle.dump(top_fitness, f)
end
