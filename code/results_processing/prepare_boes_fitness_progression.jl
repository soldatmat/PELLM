using FileIO
using BOSS
using PyCall

pickle = pyimport("pickle")
include("../de/boes/lib/boss_utils.jl")

# Define data path
data_path = joinpath(@__DIR__, "..", "de", "data", "boes", "GB1", "01")

# Load results
d = load(joinpath(data_path, "boes_wt.jld2"))
fitness_progression = get_gp_fitness_progression(d["problem"].data.Y)

# Save fitness progression
save_path = joinpath(data_path, "fitness_progression.pkl")
@pywith pybuiltin("open")(save_path, "wb") as f begin
    pickle.dump([
            fitness_progression
        ], f)
end
