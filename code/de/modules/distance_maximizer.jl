using Distances

"""
TODO

`sequences` and `embeddings` have to be in corresponding order.

`embeddings::Array{Float64}`: array with size `embedding_size`x`n_sequences`
"""
struct DistanceMaximizer <: DESilico.Mutagenesis
    sequences::Vector{Vector{Char}}
    embeddings::Array{Float64}
end

function (m::DistanceMaximizer)(parents::AbstractVector{Vector{Char}})
    parent_indexes = map(parent -> findfirst(item -> item == parent, m.sequences), parents)
    distances = pairwise(euclidean, m.embeddings, m.embeddings[:, parent_indexes])
    min_distances = map(row -> minimum(row), eachrow(distances))
    mutant_index = findmax(min_distances)[2] # TODO sample with probability derived from `min_distances`
    return [m.sequences[mutant_index]]
end
