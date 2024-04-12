using Distances

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
