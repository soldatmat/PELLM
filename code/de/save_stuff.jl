include("utils.jl")

using Serialization

serialize("test.txt", sequence_space)

ss = deserialize("test.txt")



using FileIO

data_path = joinpath(@__DIR__, "data", "neighborhoods")



using BOSS

save(
    joinpath(data_path, "gb1_esm1b_euclidean.jld2"),
    "neighborhoods", neighborhoods,
)


save(
    joinpath(data_path, "de.jld2"),
    "sequence_space", sequence_space,
    "selection_strategy", selection_strategy,
    "mutagenesis", mutagenesis,
)

d = load(joinpath(data_path, "gb1_esm1b_euclidean.jld2"))



torch.save(fp_model.state_dict(), joinpath(data_path, "TwoLayerPreceptron_state_dict.pt"))

tm = nn_model.TwoLayerPerceptron(torch.nn.Sigmoid(), embedding_extractor.embedding_size)
tm.load_state_dict(torch.load("fp_model.pt"))



results = Vector{Float64}(undef, 22000)
screened = Vector{Int}(undef, 22000)
save_period = 500
for i = 1:44
    d = load(joinpath(@__DIR__, "data", "PhoQ", "neighborhood_de", "results_$(i*save_period).jld2"))
    results[1+(i-1)*save_period:i*save_period] = d["results"]
    screened[1+(i-1)*save_period:i*save_period] = d["screened"]
end
save(
    joinpath(@__DIR__, "data", "PhoQ", "neighborhood_de", "results_1-22000.jld2"),
    "results", results,
    "screened", screened,
)

d = load(joinpath(@__DIR__, "data", "PhoQ", "neighborhood_de", "with_history", "results_sample_1000.jld2"))
results = d["results"]
screened = d["screened"]
history = d["history"]
top_variants = map(h -> reconstruct_history(h), history)
top_sequences = map(h_vector -> map(variant -> variant.sequence, h_vector), top_variants)
top_fitnesses = map(h_vector -> map(variant -> variant.fitness, h_vector), top_variants)

# ___ Python ___
using PyCall
pickle = pyimport("pickle")

file_path = joinpath(@__DIR__, "data", "PhoQ", "neighborhood_de", "with_history", "results_sample_1000_fitness_progressions.pkl")
@pywith pybuiltin("open")(file_path, "wb") as f begin
    pickle.dump([
            #results,
            #screened,
            #top_variants,
            top_fitnesses
        ], f)
end

file_path = joinpath(@__DIR__, "data", "neighborhood_de", "history_esm1b_NDYP.pkl")
@pywith pybuiltin("open")(file_path, "wb") as f begin
    pickle.dump([
            map(v -> join(extract_residues(v.sequence, mutation_positions)), sequence_space.variants),
            map(v -> join(extract_residues(v.sequence, mutation_positions)), history)
        ], f)
end

# ___ MLDE ___
folder_path = joinpath(@__DIR__, "data", "perceptron", "perceptron_distmax_finetune_01")
d = load(joinpath(folder_path, "history.jld2"))

file_path = joinpath(folder_path, "fitness_progression.pkl")
@pywith pybuiltin("open")(file_path, "wb") as f begin
    pickle.dump([
            d["top_fitness"]
        ], f)
end

file_path = joinpath(folder_path, "prediction_history.pkl")
@pywith pybuiltin("open")(file_path, "wb") as f begin
    pickle.dump([
            d["prediction_history"]
        ], f)
end

# ___ GPDE ___
d = load(joinpath(@__DIR__, "data", "PhoQ", "gpde", "gp_384_no_repeats.jld2"))

fitness_progression = get_gp_fitness_progression(problem.data.Y)

file_path = joinpath(@__DIR__, "data", "PhoQ", "gpde", "gp_384_no_repeats_fitness_progression.pkl")
@pywith pybuiltin("open")(file_path, "wb") as f begin
    pickle.dump([
            fitness_progression
        ], f)
end



# ___ Neighborhood DE ___
folder_path = joinpath(@__DIR__, "data", "PhoQ", "neighborhood_de", "with_history")
file_path = joinpath(folder_path, "wt_fitness_progression.pkl")
@pywith pybuiltin("open")(file_path, "wb") as f begin
    pickle.dump([
            fitness_progression
        ], f)
end
