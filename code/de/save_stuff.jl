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



results = Vector{Float64}(undef, 5000)
screened = Vector{Int}(undef, 5000)
for i = 1:50
    d = load(joinpath(@__DIR__, "data", "neighborhood_de", "results_distmax_$(i*100).jld2"))
    results[1+(i-1)*100:i*100] = d["results"]
    screened[1+(i-1)*100:i*100] = d["screened"]
end
save(
    joinpath(@__DIR__, "data", "neighborhood_de", "results_distmax_1-5000.jld2"),
    "results", results,
    "screened", screened,
)


# ___ Python ___
using PyCall
pickle = pyimport("pickle")

file_path = joinpath(@__DIR__, "data", "neighborhood_de", "results_distmax_1-5000.pkl")
@pywith pybuiltin("open")(file_path, "wb") as f begin
    pickle.dump([results, screened], f)
end
