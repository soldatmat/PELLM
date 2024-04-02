using DESilico
using Distances
using CSV
using DataFrames
using FileIO

include("types.jl")
include("utils.jl")
include("library_select.jl")
include("repeating_library_select.jl")
include("neighborhood_search.jl")

# ___ Function definitions ___
function _construct_neighborhoods(sequence_embeddings::AbstractMatrix{Float64})
    batch_size = 4000
    k = 100
    n_sequences = size(sequence_embeddings)[2]
    mapreduce(
        b -> _construct_neighborhoods(sequence_embeddings[:, 1+(b-1)*batch_size:b*batch_size], sequence_embeddings, k, 1 + (b - 1) * batch_size, batch_size, b),
        hcat,
        1:Int(n_sequences / batch_size),
    )
end
function _construct_neighborhoods(sequences::AbstractMatrix{Float64}, all_sequences::AbstractMatrix{Float64}, k::Int, batch_start::Int, batch_size::Int, b::Int)
    println("_construct_neighborhoods batch $(b)")
    distances = pairwise(euclidean, all_sequences, sequences)
    map(i -> distances[i+batch_start-1, i] = Inf, 1:batch_size) # set distance to self to `Inf`
    mapreduce(col -> partialsortperm(col, 1:k), hcat, eachcol(distances))
end

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
sequences_complete = _get_sequences("esm-1b_variants_complete.csv")

fitness_csv_file = "esm-1b_fitness_norm.csv"
fitness = CSV.read(joinpath(data_path, fitness_csv_file), DataFrame)
fitness = [values(row)[1] for row in eachrow(fitness)]

# ___ Screening ___
screening = DESilico.DictScreening(Dict(sequences .=> fitness), missing_fitness_value)

# ___ SelectionStrategy ___
selection_strategy = LibrarySelect(1)
#selection_strategy = RepeatingLibrarySelect()

# ___ Mutagenesis ___
#neighborhoods = _construct_neighborhoods(sequence_embeddings)
neighborhoods = load(joinpath(@__DIR__, "data", "neighborhoods", "gb1_esm1b_euclidean.jld2"))["neighborhoods"]
knn = 16
mutagenesis = NeighborhoodSearch(
    sequences_complete,
    neighborhoods[1:knn, :];
    repeat=false,
    screened=[wt_sequence],
    #n=1,
)

# ___ Change starting variant ___
#starting_mutant = "DAKQ" # "SRCG" "HECA"
#wt_sequence = collect(wt_string[1:mutation_positions[1]-1]*starting_mutant[1]*wt_string[mutation_positions[1]+1:mutation_positions[2]-1]*starting_mutant[2]*wt_string[mutation_positions[2]+1:mutation_positions[3]-1]*starting_mutant[3]*wt_string[mutation_positions[3]+1:mutation_positions[4]-1]*starting_mutant[4]*wt_string[mutation_positions[4]+1:end])

# ___ Run de! ___
wt_variant = Variant(wt_sequence, screening(wt_sequence))
sequence_space = SequenceSpace([wt_variant])

history = [sequence_space.top_variant.fitness]
for i = 1:24
    de!(
        sequence_space;
        screening,
        selection_strategy,
        mutagenesis,
        n_iterations=1,
    )
    append!(history, sequence_space.top_variant.fitness)
end

# 23 (239) for knn=16 to get 1.0 fitness
# 17 (280) for knn=24 to get 1.0 fitness
# 22 (430) for knn=32 to get 1.0 fitness
println(sequence_space.top_variant)
length(sequence_space.variants)

# ___ Plot results ___
using Plots
histogram(map(v -> v.fitness, [v for v in sequence_space.variants]), bins=range(0.0, 1.01, length=20))
plot(0:length(history)-1, history)
