"""
TODO
"""
mutable struct RepeatingLibrarySelect <: DESilico.SelectionStrategy
    library::Set{Variant}
    last_selected::Union{Variant, Nothing}

    RepeatingLibrarySelect() = new(Set{Variant}([]), nothing)
end

function (ss::RepeatingLibrarySelect)(variants::AbstractVector{Variant})
    isempty(variants) && delete!(ss.library, ss.last_selected)
    map(variant -> push!(ss.library, variant), variants)
    top_sequence = _select_top_sequence(ss)
    return [top_sequence]
end
function _select_top_sequence(ss::RepeatingLibrarySelect)
    variants = collect(ss.library)
    sort!(variants, by=x -> x.fitness, rev=true)
    top_variant = variants[1]
    ss.last_selected = copy(top_variant)
    return top_variant.sequence
end
