using StatsBase

include("types.jl")
include("utils.jl")

"""
TODO
"""
struct NeighborhoodSearch <: DESilico.Mutagenesis
    sequences::Vector{Vector{Char}}
    neighborhoods::Array{Int}
    repeat::Bool
    screened::Union{Dict{Vector{Char},Bool},Nothing}
    n::Union{Int,Nothing}

    function NeighborhoodSearch(sequences::Vector{Vector{Char}}, neighborhoods::Array{Int}, repeat::Bool, screened::Union{Dict{Vector{Char},Bool},Nothing}, n::Union{Int,Nothing})
        repeat || @assert !isnothing(screened)
        new(sequences, neighborhoods, repeat, repeat ? nothing : screened, n)
    end
end

function NeighborhoodSearch(sequences, neighborhoods; repeat::Bool=false, screened::AbstractVector{Vector{Char}}=Vector{Vector{Char}}([]), n::Union{Int,Nothing}=nothing)
    screened_dict = Dict(sequences .=> false)
    map(sequence -> screened_dict[sequence] = true, screened)
    NeighborhoodSearch(sequences, neighborhoods, repeat, screened_dict, n)
end

function (m::NeighborhoodSearch)(parents::AbstractVector{Vector{Char}})
    parent_indexes = map(parent -> findfirst(item -> item == parent, m.sequences), parents)
    mutant_indexes = mapreduce(p -> m.neighborhoods[:, p], vcat, parent_indexes)
    mutants = map(i -> m.sequences[i], mutant_indexes)
    if !m.repeat
        mutants = _filter_screened(mutants, m.screened)
    end
    if !isnothing(m.n)
        #mutants = length(mutants) <= m.n ? mutants : sample(mutants, m.n, replace=false)
        mutants = length(mutants) <= m.n ? mutants : mutants[1:m.n]
    end
    m.repeat || _update_screened!(m, mutants)
    #if length(mutants) == 0
    #    mutants = _sample_new_sequences(m)
    #end
    #println("Sending $(length(mutants)) mutants to Screening.")
    return mutants
end
_filter_screened(sequences::AbstractVector{Vector{Char}}, screened::Dict{Vector{Char},Bool}) = mapreduce(sequence -> screened[sequence] ? Vector{Vector{Char}}([]) : [sequence], vcat, sequences)
_update_screened!(m::NeighborhoodSearch, sequences::AbstractVector{Vector{Char}}) = map(sequence -> m.screened[sequence] = true, sequences)

function _sample_new_sequences(m::NeighborhoodSearch)
    pool = mapreduce(sequence -> m.screened[sequence] ? Vector{Vector{Char}}([]) : [sequence], vcat, m.sequences)
    mutants = sample(pool, 1, replace=false)
    println("Out of neighbors. Sampled $(mutants[1][mutation_positions]) $(screening([mutants[1]])).")
    return mutants
end



struct NeighborhoodSearchWithPredictor <: DESilico.Mutagenesis
    sequences::Vector{Vector{Char}}
    neighborhoods::Array{Int}
    repeat::Bool
    screened::Union{Dict{Vector{Char},Bool},Nothing}
    predictor::FitnessPredictor

    function NeighborhoodSearchWithPredictor(sequences::Vector{Vector{Char}}, neighborhoods::Array{Int}, repeat::Bool, screened::Union{Dict{Vector{Char},Bool},Nothing}, predictor::FitnessPredictor)
        repeat || @assert !isnothing(screened)
        new(sequences, neighborhoods, repeat, repeat ? nothing : screened, predictor)
    end
end

function NeighborhoodSearchWithPredictor(sequences, neighborhoods; repeat::Bool=false, screened::AbstractVector{Vector{Char}}=Vector{Vector{Char}}([]), predictor::FitnessPredictor)
    screened_dict = Dict(sequences .=> false)
    map(sequence -> screened_dict[sequence] = true, screened)
    NeighborhoodSearchWithPredictor(sequences, neighborhoods, repeat, screened_dict, predictor)
end

function (m::NeighborhoodSearchWithPredictor)(parents::AbstractVector{Vector{Char}})
    parent_indexes = map(parent -> findfirst(item -> item == parent, m.sequences), parents)
    mutant_indexes = mapreduce(p -> m.neighborhoods[:, p], vcat, parent_indexes)
    mutants = map(i -> m.sequences[i], mutant_indexes)
    if !m.repeat
        mutants = _filter_screened(mutants, m.screened)
    end
    predictions = tensor_to_matrix(m.predictor(mutants))
    mutants = [mutants[findmax(predictions)[2]]]
    m.repeat || _update_screened!(m, mutants)
    return mutants
end

_update_screened!(m::NeighborhoodSearchWithPredictor, sequences::AbstractVector{Vector{Char}}) = map(sequence -> m.screened[sequence] = true, sequences)
