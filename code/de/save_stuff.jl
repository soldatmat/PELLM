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



# ___ Python ___
using PyCall
pickle = pyimport("pickle")

file_path = joinpath(@__DIR__, "data", "neighborhood_de", "results_complete.pkl")
@pywith pybuiltin("open")(file_path, "wb") as f begin
    pickle.dump([d["results"], d["screened"]], f)
end
