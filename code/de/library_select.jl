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

function (ss::LibrarySelect)(variants::AbstractVector{Variant})
    map(variant -> push!(ss.library, variant), variants)
    return _select_top_k!(ss)
end
function _select_top_k!(ss::LibrarySelect)
    variants = collect(ss.library)
    sort!(variants, by=x -> x.fitness, rev=true)
    _select_first_k!(ss, variants)
end
function _select_first_k!(ss::LibrarySelect, variants::AbstractVector{Variant})
    selection = variants[1:ss.k]
    map(variant -> delete!(ss.library, variant), selection)
    map(variant -> variant.sequence, selection)
end

(ss::LibrarySelect)() = ss(Vector{Variant}([]))
