"""
TODO
"""
struct CumulativeSelect <: DESilico.SelectionStrategy
    library::Set{Vector{Char}}
end

CumulativeSelect() = CumulativeSelect(Set{Variant}([]))
CumulativeSelect(library::Vector{Vector{Char}}) = CumulativeSelect(Set(library))

function (ss::CumulativeSelect)(variants::AbstractVector{Variant})
    map(variant -> push!(ss.library, variant.sequence), variants)
    return collect(ss.library)
end
