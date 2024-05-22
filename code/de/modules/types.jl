"""
Model trained to predict fitness from sequence.

Structures derived from this type have to implement the following methods:

`(::CustomFitnessPredictor)(sequences::AbstractVector{Vector{Char}})`
This method should return the sequences' predicted fitness values as a subtype of `AbstarctVector{Float64}`.

`train(model::CustomFitnessPredictor, variants::AbstractVector{Variant})`
This method should use the provided `variants` to train the `model`.
"""
abstract type FitnessPredictor end

"""
Model used to extract embedding from sequence.

Structures derived from this type need to have the following field:

- `embedding_size::Int`: Size of the sequence embedding.

Structures derived from this type have to implement the following methods:

`(::CustomEmbeddingExtractor)(sequences::AbstractVector{Vector{Char}})`
This method should return sequence embeddings of `sequences` as a subtype of `AbstractVector{Vector{Float64}}`.
"""
abstract type AbstractEmbeddingExtractor end
