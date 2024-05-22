using StatsBase

"""
TODO
"""
struct LibrarySelect <: DESilico.SelectionStrategy
    k::Int
    library::Set{Variant}

    function LibrarySelect(k::Int, library::Set{Variant})
        k > 0 || throw(ArgumentError("`k` needs to be greater than 0"))
        new(k, library)
    end
end

LibrarySelect(k::Int) = LibrarySelect(k, Set{Variant}([]))
LibrarySelect(k::Int, library::AbstractVector{Variant}) = LibrarySelect(k, Set(library))

function (ss::LibrarySelect)(variants::AbstractVector{Variant})
    map(variant -> push!(ss.library, variant), variants)
    _select_top_k(ss)
end

"""
TODO
"""
struct SamplingLibrarySelect <: DESilico.SelectionStrategy
    k::Int
    library::Set{Variant}
    top_fitness::Vector{Float64}
    distance_maximizer::DistanceMaximizer
    screening::DESilico.Screening # TODO remove
    sequence_space::SequenceSpace

    function SamplingLibrarySelect(k::Int, library::Set{Variant}, distance_maximizer::DistanceMaximizer, screening::DESilico.Screening, sequence_space::SequenceSpace)
        k > 0 || throw(ArgumentError("`k` needs to be greater than 0"))
        @assert length(library) > 0
        variants = collect(library)
        sort!(variants, by=x -> x.fitness, rev=true)
        top_fitness = vec(zeros(51, 1))
        top_fitness[1] = -1
        top_fitness[end] = variants[1].fitness
        new(k, library, top_fitness, distance_maximizer, screening, sequence_space)
    end
end

SamplingLibrarySelect(k::Int, library::AbstractVector{Variant}, distance_maximizer::DistanceMaximizer, screening::DESilico.Screening, sequence_space::SequenceSpace) = SamplingLibrarySelect(k, Set(library), distance_maximizer, screening, sequence_space)

function (ss::SamplingLibrarySelect)(variants::AbstractVector{Variant})
    map(variant -> _push_variant!(ss, variant), variants)
    println("pushed variants: $(length(variants)), length(library) = $(length(ss.library))")
    if findmax(ss.top_fitness)[2] == 1
        selection = _sample_k(ss)
    else
        selection = _select_top_k(ss)
    end
    return selection
end
function _push_variant!(ss::SamplingLibrarySelect, variant::Variant)
    push!(ss.library, variant)
    push!(ss.top_fitness, variant.fitness > ss.top_fitness[end] ? variant.fitness : ss.top_fitness[end])
    popfirst!(ss.top_fitness)
end

function _select_top_k(ss::Union{LibrarySelect,SamplingLibrarySelect})
    variants = collect(ss.library)
    sort!(variants, by=x -> x.fitness, rev=true)
    map(variant -> delete!(ss.library, variant), variants[1:ss.k])
    map(variant -> variant.sequence, variants[1:ss.k])
end

function _sample_k(ss::Union{LibrarySelect,SamplingLibrarySelect})
    println("SAMPLE")
    map(v -> push!(ss.library, v), ss.sequence_space.variants)
    variants = collect(ss.library)
    starting_sequences = Vector{Vector{Char}}(undef, ss.k)
    for i = 1:ss.k
        starting_sequences[i] = ss.distance_maximizer(map(v -> v.sequence, variants))[1]
        push!(variants, Variant(starting_sequences[i], ss.screening(starting_sequences[i])))
    end
    map(v -> delete!(ss.library, v), collect(ss.library))
    map(i -> ss.top_fitness[i] = 0.0, eachindex(ss.top_fitness))
    ss.top_fitness[1] = -1
    return starting_sequences
end

(ss::Union{LibrarySelect,SamplingLibrarySelect})() = (ss)(Vector{Variant}([]))
