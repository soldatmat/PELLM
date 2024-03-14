"""
Large language model trained on large amount of sequences.

Structures derived from this type have the following fields:

- `model::PyObject`: `torch.nn` model.
- `embedding_size::Int`: Size of the sequence embedding.
- `device::PyObject`: `torch.device` on which the model is loaded.

Structures derived from this type can implement the following methods:

    `tokenize(llm::ESM1b, sequence::AbstractVector{Char})`

Return tokenized version of `sequence` as a tensor ready to be inputted into `llm`.

    `train!(llm::CustomLLM, sequences::AbstractVector{Vector{Char}})`

Use the provided `sequences` to train the `model`.
This method can have additional keyword arguments based on the LLM's capabilities.

# Keywords
- `mask_positions::AbstractVector{Int}`: If provided, enables MLM training mode with masks at `mask_positions`.

    `extract_embedding(llm::CustomLLM, batch::PyObject)`

Takes a torch tensor with tokenized sequences and returns embeddings as a tensor
with size (length of `batch`, length of longest sequence in `batch`, embedding size) on the same device.

    `extract_sequence_embedding(llm::ESM1b, batch::PyObject)`

Takes a torch tensor with tokenized sequences and returns sequence embeddings as a tensor
with size (length of `batch`, embedding size) on the same device.

    `extract_contextualized_embeding(llm::ESM1b, batch::PyObject, token_index::Int)`

Takes a torch tensor with tokenized sequences and returns embeddings at `token_index` as a tensor
with size (length of `batch`, embedding size) on the same device.

    `get_pseudolikelihoods(llm::CustomLLM, masked_sequence::Vector{Float32}, mask_positions::Vector{Int})`

Returns pseudolikelihoods of each symbol for each mask in `masked_sequence`
as `Vector{Vector{Tuple{Char, Float32}}}` with length equal to length of `mask_positions`.

    `mask_sequence!(llm::ESM1b, sequence::PyObject, mask_index::Int)`

Sets token at `mask_index` in a tokenized `sequence` to the mask token used by `llm`.
"""
abstract type LLM end

tokenize(llm::LLM, sequence::String) = tokenize(llm, collect(sequence))
tokenize(llm::LLM, sequences::AbstractVector{Vector{Char}}) = map(s -> tokenize(llm, s), sequences)
tokenize(llm::LLM, sequences::AbstractVector{String}) = tokenize(llm, map(s -> collect(s), sequences))

mask_sequence!(llm::LLM, sequence::PyObject, mask_indexes::AbstractVector{Int}) = map(mask_index -> mask_sequence!(llm, sequence, mask_index), mask_indexes)
