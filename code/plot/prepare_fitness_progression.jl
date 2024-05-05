using FileIO
using DESilico
using BOSS
using PyCall

pickle = pyimport("pickle")

include("../de/utils.jl")

data_path = joinpath(@__DIR__, "..", "de", "data", "perceptron_distmax_01")

history = load(joinpath(data_path, "history.jld2"))
fitness_progression = history["top_fitness"]

save_path = joinpath(data_path, "fitness_progression.pkl")
@pywith pybuiltin("open")(save_path, "wb") as f begin
    pickle.dump([
            fitness_progression
        ], f)
end
