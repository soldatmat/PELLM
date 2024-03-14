"""
Uses provided Variants to train a fitness predictor and returns `k` sequences
with highest predicted fitness from a pre-defined pool of sequences.

    TopKPredicted(model::FitnessPredictor, k::Int, sequences::Vector{Vector{Char}})

# Arguments
- `model::FitnessPredictor`: Model which predicts fitness from sequence.
- `k::Int`: Defines the number of sequences which will be selected.
- `sequences::Vector{Vector{Char}}`: Pool of sequences from which `k` sequences with highest predicted fitness are chosen.

    TopKPredicted(model::FitnessPredictor, k::Int, sequence_length::Int, alphabet::Set{Char})

# Arguments
- `model::FitnessPredictor`: Model which predicts fitness from sequence.
- `k::Int`: Defines the number of sequences which will be selected.
- `sequence_length::Int`: Length of `sequences` which will be constructed from `alphabet`.
- `alphabet::Set{Char}`: Alphabet used to construct `sequences`.
"""
struct TopKPredicted{T<:FitnessPredictor} <: DESilico.SelectionStrategy
    model::T
    k::Int
    sequences::Vector{Vector{Char}}

    function TopKPredicted(model::T, k::Int, sequences::Vector{Vector{Char}}) where {T<:FitnessPredictor}
        k > 0 || throw(ArgumentError("`k` needs to be greater than 0"))
        @assert k <= length(sequences)
        new{T}(model, k, sequences)
    end
end

TopKPredicted(model, k, sequence_length::Int, alphabet::Set{Char}) = TopKPredicted(model, k, _recombine(sequence_length, alphabet))

_recombine(sequence_length::Int, alphabet::Set{Char}) = map(sequence -> collect(sequence), collect(Iterators.product(ntuple(_ -> alphabet, sequence_length)...))[:])

# TODO implement loading pool of sequences from file
#TopKPredicted(k, model, path::String) = TopKPredicted(k, model, _load_sequences(path))

function (ss::TopKPredicted)(variants::AbstractVector{Variant})
    train!(ss.model, variants)
    prediction = ss.model(ss.sequences)
    evaluated_variants = map((s, f) -> Variant(s, f), ss.sequences, prediction)
    sort!(evaluated_variants, by=x -> x.fitness, rev=true)
    selection = _select_top_k(ss, evaluated_variants)
    return selection
end

# TODO move into utils, unified for TopK and TopKPredicted
function _select_top_k(ss::TopKPredicted, variants::AbstractVector{Variant})
    selection = Vector{Vector{Char}}(undef, ss.k)
    for (idx, variant) in enumerate(variants[1:ss.k])
        selection[idx] = variant.sequence
    end
    return selection
end
