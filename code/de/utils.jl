using PyCall
using DataFrames

torch = pyimport("torch")

function _tensor_probabilities_to_touples(probabilities::PyObject, alphabet::Vector{Char})
    probabilities = probabilities.cpu().numpy() # PyObject tensor -> Matrix{Float32}
    probabilities = [probabilities[seq, :] for seq in 1:size(probabilities)[1]] # Matrix{Float32} -> Vector{Vector{Float32}}
    [[(alphabet[symbol], pdist[symbol]) for symbol in eachindex(alphabet)] for pdist in probabilities] # Vector{Vector{Float32}} -> Vector{Dict{Char, Float32}}
end

default_torch_device() = torch.device(torch.cuda.is_available() ? "cuda" : "cpu")

tensor_to_matrix(tensor::PyObject) = tensor.cpu().numpy()

ensure_tensor(data::Any) = pybuiltin("type")(data) == torch.Tensor ? data : torch.tensor(data)

recombine_symbols(sequence_length::Int, alphabet::Set{Char}) = map(sequence -> collect(sequence), collect(Iterators.product(ntuple(_ -> alphabet, sequence_length)...))[:])

function _get_variants(data_path::String, csv_file::String)
    variants = CSV.read(joinpath(data_path, csv_file), DataFrame)
    [collect(values(row)[1]) for row in eachrow(variants)]
end
_construct_sequence(variant::Vector{Char}, wt_string::String, mutation_positions::Vector{Int}) = collect(wt_string[1:mutation_positions[1]-1] * variant[1] * wt_string[mutation_positions[1]+1:mutation_positions[2]-1] * variant[2] * wt_string[mutation_positions[2]+1:mutation_positions[3]-1] * variant[3] * wt_string[mutation_positions[3]+1:mutation_positions[4]-1] * variant[4] * wt_string[mutation_positions[4]+1:end])
_construct_sequences(variants::AbstractVector{Vector{Char}}, wt_string::String, mutation_positions::Vector{Int}) = map(v -> _construct_sequence(v, wt_string, mutation_positions), variants)
_get_sequences(data_path::String, csv_file::String, wt_string::String, mutation_positions::Vector{Int}) = _construct_sequences(_get_variants(data_path, csv_file), wt_string, mutation_positions)

function _get_fitness(data_path::String, csv_file::String)
    fitness = CSV.read(joinpath(data_path, csv_file), DataFrame)
    fitness = [values(row)[1] for row in eachrow(fitness)]
end

function _get_sequence_emebeddings(data_path::String, csv_file::String)
    sequence_embeddings = CSV.read(joinpath(data_path, csv_file), DataFrame)
    sequence_embeddings = [collect(values(row)) for row in eachrow(sequence_embeddings)]
end

extract_residues(sequence::AbstractVector{Char}, mutation_posisitions::AbstractVector{Int}) = map(pos -> sequence[pos], mutation_posisitions)

function reconstruct_history(variants::AbstractVector{Variant})
    top_variant = Vector{Variant}(undef, length(variants))
    top_variant[1] = variants[1]
    map(i -> variants[i].fitness > top_variant[i-1].fitness ? top_variant[i] = variants[i] : top_variant[i] = top_variant[i-1], 2:length(variants))
    return top_variant
end
