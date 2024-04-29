using Distances

"""
TODO
"""
struct PredictionDistanceMaximizer{T<:FitnessPredictor} <: DESilico.SelectionStrategy
    model::T
    sequences::Vector{Vector{Char}}
    labels::Vector{Float64}
    k::Int
    repeat::Bool
    train_callback::Function

    function PredictionDistanceMaximizer(model::T, sequences::Vector{Vector{Char}}, screened::AbstractVector{Variant}, k::Int, repeat::Bool, train_callback::Function) where {T<:FitnessPredictor}
        k > 0 || throw(ArgumentError("`k` needs to be greater than 0"))
        @assert k <= length(sequences)
        screened_sequences = map(variant -> variant.sequence, screened)
        sequences = filter(sequence -> !(sequence in screened_sequences), sequences)
        labels = map(variant -> variant.fitness, screened)
        new{T}(model, sequences, labels, k, repeat, train_callback)
    end
end

PredictionDistanceMaximizer(model, sequences; screened::AbstractVector{Variant}=Vector{Variant}([]), k::Int=1, repeat::Bool=true, train_callback::Function=()->nothing) = PredictionDistanceMaximizer(model, sequences, screened, k, repeat, train_callback)
PredictionDistanceMaximizer(model, sequence_length::Int, alphabet::Set{Char}; screened::AbstractVector{Variant}=Vector{Variant}([]), k::Int=1, repeat::Bool=true, train_callback::Function=()->nothing) = PredictionDistanceMaximizer(model, recombine_symbols(sequence_length, alphabet), screened, k, repeat, train_callback)

function (ss::PredictionDistanceMaximizer)(variants::AbstractVector{Variant})
    ss.repeat || _update_sequences!(ss, variants)
    train!(ss.model, variants)
    ss.train_callback(ss)
    prediction = ss.model(ss.sequences)
    predicted_variants = map((s, f) -> Variant(s, f), ss.sequences, prediction)
    _select_farthest_k(ss, predicted_variants)
end

function _update_sequences!(ss::PredictionDistanceMaximizer, variants::AbstractVector{Variant})
    append!(map(variant -> variant.fitness, variants), ss.labels)
    screened_sequences = map(variant -> variant.sequence, variants)
    filter!(sequence -> !(sequence in screened_sequences), ss.sequences)
end

function _select_farthest_k(ss::PredictionDistanceMaximizer, predicted_variants::AbstractVector{Variant})
    min_distances = _get_min_distances(ss, predicted_variants)
    pairs = map(i -> (predicted_variants[i].sequence, min_distances[i]), eachindex(predicted_variants))
    sort!(pairs, by=pair -> pair[2], rev=true)
    map(pair -> pair[1], pairs[1:ss.k])
end
function _get_min_distances(ss::PredictionDistanceMaximizer, predicted_variants::AbstractVector{Variant})
    distances = pairwise(cityblock, map(v -> v.fitness, predicted_variants), ss.labels)
    min_distances = map(row -> minimum(row), eachrow(distances))
end
