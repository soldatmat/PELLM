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
include("top_k_predicted.jl")
include("llm_sampler.jl")

GC.gc() # For re-runs, GPU allocs by Python sometimes do not get freed automatically

# Data specific parameters:
alphabet = DESilico.alphabet.protein
data_path = joinpath(@__DIR__, "..", "..", "data", "GB1")
xlxs_file = "elife-16965-supp1.xlsx"
wt_string = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"  # ['V', 'D', 'G', 'V']
wt_sequence = collect(wt_string)
mutation_positions = [39, 40, 41, 54]
missing_fitness_value = 0.0

seq_embedding_csv_file = "esm-1b_embedding_complete.csv"
sequence_embeddings = CSV.read(joinpath(data_path, seq_embedding_csv_file), DataFrame)
sequence_embeddings = [collect(values(row)) for row in eachrow(sequence_embeddings)]

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

# ___ LLM ___
llm = ESM1b()
mask_token = llm.alphabet.mask_idx
mask_string = llm.alphabet.all_toks[llm.alphabet.mask_idx+1] # +1 for julia vs Python indexing

# ___ Screening ___
# TODO use normalized fitness
# TODO implement loading from pandas dataframe
#screening = DESilico.DictScreening(joinpath(data_path, xlxs_file), missing_fitness_value)
screening = DESilico.DictScreening(Dict(sequences .=> fitness), missing_fitness_value)

# ___ SelectionStrategy ___
#embedding_extractor = LLMEmbeddingExtractor(llm; return_tensor=true)
embedding_extractor = DictEmbeddingExtractor(Dict(sequences_complete .=> sequence_embeddings))
fp_model = nn_model.TwoLayerPerceptron(torch.nn.Sigmoid(), embedding_extractor.embedding_size)
fitness_predictor = EmbeddingNN(embedding_extractor, fp_model)
#selection_strategy = TopKPredicted(fitness_predictor, 8000, length(variants[1]), alphabet)
selection_strategy = TopKPredicted(fitness_predictor, 100, sequences_complete) # TODO k = 8000 instead of 100

# ___ Mutagenesis ___
alphabet_extractor = LLMSampler(llm; sampling_sequence=wt_sequence, alphabet, k=3) # k=3 from AFP-DE
mutagenesis = DESilico.Recombination(alphabet_extractor; mutation_positions, n=nothing) # n=24 from AFP-DE

# ___ Run de! ___
wt_variant = Variant(wt_sequence, screening(wt_sequence))
sequence_space = SequenceSpace([wt_variant])
de!(
    sequence_space;
    screening,
    selection_strategy,
    mutagenesis,
    n_iterations=2,
)
println(sequence_space.top_variant)

# ___ Plot results ___
using Plots
histogram(map(v->v.fitness, [v for v in sequence_space.variants]), bins=range(0,1,length=20))
