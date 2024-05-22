using PyCall

torch = pyimport("torch")

pushfirst!(pyimport("sys")."path", joinpath(@__DIR__, "..", "python"))
llm_embedding_extractor = pyimport("llm_embedding_extractor")

"""
Extracts sequence embeddings using a specified `LLM` (Large Language Model).

    LLMEmbeddingExtractor(llm::LLM, embedding_size::Int, batch_size::Int, return_tensor::Bool)
    LLMEmbeddingExtractor(llm::LLM; embedding_size::Int=llm.embedding_size, batch_size::Int=1, return_tensor::Bool=false)

# Arguments
- `llm::LLM`: The `LLM` used to extract embeddings.
- `embedding_size::Int`: Size of sequence embeddings extracted by `llm`.
- `batch_size::Int`: Size of batches used in embedding extraction.
- `return_tensor::Bool`: If `true`, the outputed `torch.tensor` with embeddings is not converted into `Vector{Vector{Int}}`.
"""
struct LLMEmbeddingExtractor{T<:LLM} <: AbstractEmbeddingExtractor
    llm::T
    embedding_size::Int
    batch_size::Int
    return_tensor::Bool
    LLMEmbeddingExtractor(llm::T, embedding_size::Int, batch_size::Int, return_tensor::Bool) where {T<:LLM} = new{T}(llm, embedding_size, batch_size, return_tensor)
end

LLMEmbeddingExtractor(llm; embedding_size=llm.embedding_size, batch_size=1, return_tensor=false) = LLMEmbeddingExtractor(llm, embedding_size, batch_size, return_tensor)

# TODO rewrite in Python with DataLoader
function (ee::LLMEmbeddingExtractor)(sequences::AbstractVector{Vector{Char}})
    println("LLMEmbeddingExtractor extract embeddings")
    ee.llm.model.eval()
    torch.set_grad_enabled(false)

    batch_borders = _get_batch_borders(length(sequences), ee.batch_size)
    embedding = Vector{PyObject}(undef, length(batch_borders) - 1)
    tokenized_sequences = tokenize(llm, sequences).to(llm.device)
    for b in 1:length(batch_borders)-1
        batch = tokenized_sequences.narrow(0, batch_borders[b], batch_borders[b+1] - batch_borders[b])
        embedding[b] = extract_sequence_embedding(ee.llm, batch)
        #GC.gc() # GPU allocs by Python do not get freed automatically in some iterations
    end
    embedding = torch.cat(embedding)
    if !ee.return_tensor
        embedding = tensor_to_matrix(embedding)
    end

    torch.set_grad_enabled(true)
    return embedding
end
function (ee::LLMEmbeddingExtractor{ESM1b})(sequences::AbstractVector{Vector{Char}})
    tokenized_sequences = tokenize(llm, sequences).to(llm.device)
    llm_embedding_extractor.extract_embeddings(ee.llm.model, tokenized_sequences, ee.embedding_size, ee.batch_size)
end

function _get_batch_borders(n_sequences::Int, batch_size::Int)
    n_full_batches = Int(floor(n_sequences / batch_size))
    batch_borders = [i * batch_size for i in 0:n_full_batches]
    (n_sequences % batch_size == 0) || append!(batch_borders, n_sequences)
    return batch_borders
end
