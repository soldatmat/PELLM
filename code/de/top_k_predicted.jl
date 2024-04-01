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
    sequences::Vector{Vector{Char}}
    k::Int
    repeat::Bool
    screened::Union{Dict{Vector{Char}, Bool}, Nothing}

    function TopKPredicted(model::T, sequences::Vector{Vector{Char}}, k::Int, repeat::Bool) where {T<:FitnessPredictor}
        k > 0 || throw(ArgumentError("`k` needs to be greater than 0"))
        @assert k <= length(sequences)
        screened = repeat ? nothing : Dict(sequences .=> false)
        new{T}(model, sequences, k, repeat, screened)
    end
end

TopKPredicted(model, sequences; k::Int=1, repeat::Bool=true) = TopKPredicted(model, sequences, k, repeat)
TopKPredicted(model, sequence_length::Int, alphabet::Set{Char}; k::Int=1, repeat::Bool=true) = TopKPredicted(model, recombine_symbols(sequence_length, alphabet), k, repeat)

# TODO implement loading pool of sequences from file
#TopKPredicted(k, model, path::String) = TopKPredicted(k, model, _load_sequences(path))

function (ss::TopKPredicted)(variants::AbstractVector{Variant})
    ss.repeat || _update_screened!(ss, variants)
    train!(ss.model, variants)
    prediction = ss.model(ss.sequences)
    predicted_variants = map((s, f) -> Variant(s, f), ss.sequences, prediction)
    _select_top_k!(ss, predicted_variants)
end

_update_screened!(ss::TopKPredicted, variants::AbstractVector{Variant}) = map(variant -> ss.screened[variant.sequence] = true, variants)

function _select_top_k!(ss::TopKPredicted, variants::AbstractVector{Variant})
    sort!(variants, by=x -> x.fitness, rev=true)
    selection = ss.repeat ? _select_first_k(ss, variants) : _select_first_k_without_repeats!(ss, variants)
    return selection
end

function _select_first_k(ss::TopKPredicted, variants::AbstractVector{Variant})
    selection = Vector{Vector{Char}}(undef, ss.k)
    for (idx, variant) in enumerate(variants[1:ss.k])
        selection[idx] = variant.sequence
    end
    return selection
end

function _select_first_k_without_repeats!(ss::TopKPredicted, variants::AbstractVector{Variant})
    selection = Vector{Vector{Char}}(undef, ss.k)
    n_selected = 0
    for variant in variants
        ss.screened[variant.sequence] && continue
        n_selected += 1
        selection[n_selected] = variant.sequence
        n_selected == ss.k && break
    end
    if n_selected < ss.k
        println("Warning: TopKPredicted selected only $(n_selected) sequences but k is $(ss.k).")
        return selection[1:n_selected]
    end
    return selection
end
