using DESilico
using Distances
using CSV
using DataFrames
using FileIO

include("types.jl")
include("utils.jl")
include("library_select.jl")
include("neighborhood_search.jl")

# ___ Data specific parameters ___
data_path = joinpath(@__DIR__, "..", "..", "data", "GB1")
wt_string = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"  # ['V', 'D', 'G', 'V']
wt_sequence = collect(wt_string)
mutation_positions = [39, 40, 41, 54]
missing_fitness_value = 0.0

seq_embedding_csv_file = "esm-1b_embedding_complete.csv"
sequence_embeddings = CSV.read(joinpath(data_path, seq_embedding_csv_file), DataFrame)
sequence_embeddings = Matrix{Float64}(sequence_embeddings)
sequence_embeddings = transpose(sequence_embeddings)

function _get_variants(csv_file::String)
    variants = CSV.read(joinpath(data_path, csv_file), DataFrame)
    [collect(values(row)[1]) for row in eachrow(variants)]
end
_construct_sequence(variant::Vector{Char}) = collect(wt_string[1:mutation_positions[1]-1] * variant[1] * wt_string[mutation_positions[1]+1:mutation_positions[2]-1] * variant[2] * wt_string[mutation_positions[2]+1:mutation_positions[3]-1] * variant[3] * wt_string[mutation_positions[3]+1:mutation_positions[4]-1] * variant[4] * wt_string[mutation_positions[4]+1:end])
_construct_sequences(variants::AbstractVector{Vector{Char}}) = map(v -> _construct_sequence(v), variants)
_get_sequences(csv_file::String) = _construct_sequences(_get_variants(csv_file))

sequences = _get_sequences("esm-1b_variants.csv")
variants_complete = _get_variants("esm-1b_variants_complete.csv")
sequences_complete = _construct_sequences(variants_complete)

fitness_csv_file = "esm-1b_fitness_norm.csv"
fitness = CSV.read(joinpath(data_path, fitness_csv_file), DataFrame)
fitness = [values(row)[1] for row in eachrow(fitness)]

neighborhoods = load(joinpath(@__DIR__, "data", "neighborhoods", "gb1_esm1b_euclidean.jld2"))["neighborhoods"]

# ___ Screening ___
screening = DESilico.DictScreening(Dict(sequences .=> fitness), missing_fitness_value)

results = Vector{Float64}(undef, length(variants_complete))
screened = Vector{Int}(undef, length(variants_complete))
for (v, variant) in enumerate(variants_complete)
    (v % 100 == 0) && println("$(v)/$(length(variants_complete))")

    # ___ SelectionStrategy ___
    selection_strategy = LibrarySelect(1)

    # ___ Mutagenesis ___
    knn = 16
    mutagenesis = NeighborhoodSearch(
        sequences_complete,
        neighborhoods[1:knn, :];
        repeat=false,
        screened=[wt_sequence],
    )

    # ___ Change starting sequence ___
    starting_sequence = _construct_sequence(variant)

    # ___ Run de! ___
    wt_variant = Variant(starting_sequence, screening(starting_sequence))
    sequence_space = SequenceSpace([wt_variant])
    de!(
        sequence_space;
        screening,
        selection_strategy,
        mutagenesis,
        n_iterations=24,
    )

    # ___ Save results ___
    results[v] = sequence_space.top_variant.fitness
    screened[v] = length(sequence_space.variants)
end

save(
    joinpath(@__DIR__, "data", "neighborhood_de", "results_complete.jld2"),
    "results", results,
    "screened", screened,
)
