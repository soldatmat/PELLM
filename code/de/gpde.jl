using DESilico
using CSV
using DataFrames
using PyCall

torch = pyimport("torch")

pushfirst!(pyimport("sys")."path", joinpath(@__DIR__))
nn_model = pyimport("two_layer_perceptron")

include("types.jl")
include("llm.jl")
include("utils.jl")
include("embedding_nn.jl")
include("llm_embedding_extractor.jl")
include("dict_embedding_extractor.jl")
include("esm1b.jl")
include("gaussian_process_selection.jl")

# Data specific parameters:
alphabet = DESilico.alphabet.protein
data_path = joinpath(@__DIR__, "..", "..", "data", "GB1")
xlxs_file = "elife-16965-supp1.xlsx"
wt_string = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"  # ['V', 'D', 'G', 'V']
wt_sequence = collect(wt_string)
mutation_positions = [39, 40, 41, 54]
missing_fitness_value = 0.0

# ___ Load ___
function _get_sequences(csv_file::String)
    variants = CSV.read(joinpath(data_path, csv_file), DataFrame)
    variants = [collect(values(row)[1]) for row in eachrow(variants)]
    map(v -> collect(wt_string[1:mutation_positions[1]-1] * v[1] * wt_string[mutation_positions[1]+1:mutation_positions[2]-1] * v[2] * wt_string[mutation_positions[2]+1:mutation_positions[3]-1] * v[3] * wt_string[mutation_positions[3]+1:mutation_positions[4]-1] * v[4] * wt_string[mutation_positions[4]+1:end]), variants)
end
sequences = _get_sequences("esm-1b_variants.csv")
sequences_complete = _get_sequences("esm-1b_variants_complete.csv")

fitness_csv_file = "esm-1b_fitness_norm.csv"
fitness = CSV.read(joinpath(data_path, fitness_csv_file), DataFrame)
fitness = [values(row)[1] for row in eachrow(fitness)]

seq_embedding_csv_file = "esm-1b_embedding_complete.csv"
sequence_embeddings = CSV.read(joinpath(data_path, seq_embedding_csv_file), DataFrame)
sequence_embeddings = [collect(values(row)) for row in eachrow(sequence_embeddings)]

# ___ Run ___
screening = DESilico.DictScreening(Dict(sequences .=> fitness), missing_fitness_value)

# TODO selection_strategy = GaussianProcessSelection
embedding_extractor = DictEmbeddingExtractor(Dict(sequences_complete .=> sequence_embeddings))
fp_model = nn_model.TwoLayerPerceptron(torch.nn.Sigmoid(), embedding_extractor.embedding_size)
fitness_predictor = EmbeddingNN(embedding_extractor, fp_model)
selection_strategy = GaussianProcessSelection(fitness_predictor, sequences_complete)

struct NoMutagenesis <: DESilico.Mutagenesis end
(::NoMutagenesis)(parents::AbstractVector{Vector{Char}}) = parents
mutagenesis = NoMutagenesis()

wt_variant = Variant(wt_sequence, screening(wt_sequence))
sequence_space = SequenceSpace([wt_variant])
de!(
    sequence_space;
    screening,
    selection_strategy,
    mutagenesis,
    n_iterations=34,
)
map(variant -> (variant.sequence[mutation_positions], variant.fitness), collect(sequence_space.variants))
maximum(map(variant -> variant.fitness, collect(sequence_space.variants)))
