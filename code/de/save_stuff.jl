using Serialization

serialize("test.txt", sequence_space)

ss = deserialize("test.txt")



using FileIO

data_path = joinpath(@__DIR__, "data", "neighborhoods")


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
top_variants = map(h_vector -> map(variant -> variant.sequence, h_vector), history)
top_fitnesses = map(h_vector -> map(variant -> variant.fitness, h_vector), history)

# ___ Python ___
using PyCall
pickle = pyimport("pickle")

file_path = joinpath(@__DIR__, "data", "PhoQ", "neighborhood_de", "with_history", "results_sample_1000.pkl")
@pywith pybuiltin("open")(file_path, "wb") as f begin
    pickle.dump([
            results,
            screened,
            #top_variants,
            #top_fitnesses
        ], f)
end

file_path = joinpath(@__DIR__, "data", "neighborhood_de", "history_esm1b_NDYP.pkl")
@pywith pybuiltin("open")(file_path, "wb") as f begin
    pickle.dump([
            map(v -> join(extract_residues(v.sequence)), sequence_space.variants),
            map(v -> join(extract_residues(v.sequence)), history)
        ], f)
end
