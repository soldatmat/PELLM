"""
Returns sequence embeddings from a `Dict`.

    DictEmbeddingExtractor(dict::Dict{Vector{Char}, Vector{Int}}, embedding_size::Int)
    DictEmbeddingExtractor(dict::Dict{Vector{Char}, Vector{Int}}; embedding_size::Int=length([v for v in values(dict)][1]))

# Arguments
- `dict::Dict{Vector{Char}, Vector{Float64}}`: `Dict` with sequences as keys and sequence embeddings as values.
- `embedding_size::Int`: Size of sequence embeddings in `dict`.
"""
struct DictEmbeddingExtractor <: AbstractEmbeddingExtractor
    dict::Dict{Vector{Char}, Vector{Float64}}
    embedding_size::Int
end

DictEmbeddingExtractor(dict; embedding_size=length([v for v in values(dict)][1])) = DictEmbeddingExtractor(dict, embedding_size)

(ee::DictEmbeddingExtractor)(sequences::AbstractVector{Vector{Char}}) = map(sequence -> ee.dict[sequence], sequences)
