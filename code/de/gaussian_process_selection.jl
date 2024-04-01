using GaussianProcesses

"""
TODO
"""
struct GaussianProcessSelection{T<:FitnessPredictor} <: DESilico.SelectionStrategy
    model::T
    k::Int
    sequences::Vector{Vector{Char}}
    variants::Vector{Variant}

    function GaussianProcessSelection(model::T, k::Int, sequences::Vector{Vector{Char}}, variants::Vector{Variant}) where {T<:FitnessPredictor}
        k > 0 || throw(ArgumentError("`k` needs to be greater than 0"))
        @assert k <= length(sequences)
        new{T}(model, k, sequences, variants)
    end
end

GaussianProcessSelection(model, sequences::Vector{Vector{Char}}; k::Int=1, variants::Vector{Variant}=Vector{Variant}([])) = GaussianProcessSelection(model, k, sequences, variants)
GaussianProcessSelection(model, sequence_length::Int, alphabet::Set{Char}; k::Int=1, variants::Vector{Variant}=Vector{Variant}([])) = GaussianProcessSelection(model, k, _recombine(sequence_length, alphabet), variants)

function (ss::GaussianProcessSelection)(variants::AbstractVector{Variant})
    gp = _update_distribution(ss, variants)
    _select_targets(ss, gp)
end

function _update_distribution(ss::GaussianProcessSelection, variants::AbstractVector{Variant})
    append!(ss.variants, variants)

    alphabet = DESilico.alphabet.protein
    symbol_to_float = Dict(collect(alphabet) .=> 1:length(alphabet)) # TODO move
    x = hcat(mapreduce(variant -> map(symbol -> symbol_to_float[symbol], variant.sequence), hcat, ss.variants))
    y = map(variant -> variant.fitness, ss.variants)

    mean = MeanConst(0.0)
    #kernel = exponentiated_quadratic(1)
    kernel = LinIso(4.0)
    gp = GPE(x, y, mean, kernel)
end

function _select_targets(ss::GaussianProcessSelection, gp)
    symbol_to_float = Dict(collect(alphabet) .=> 1:length(alphabet)) # TODO move
    targets = float(mapreduce(sequence -> map(symbol -> symbol_to_float[symbol], sequence), hcat, ss.sequences))
    mean, variance = predict_f(gp, targets; full_cov=false)
    [ss.sequences[argmax(variance)]] # TODO return top ss.k, not just top one
end

exponentiated_quadratic(sigma) = (xa, xb) -> exp((-1 / (2 * sigma^2)) * sum(xa .!= xb)^2)
