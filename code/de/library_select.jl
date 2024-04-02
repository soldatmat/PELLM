"""
TODO
"""
struct LibrarySelect <: DESilico.SelectionStrategy
    k::Int
    library::Set{Variant}

    function LibrarySelect(k::Int)
        k > 0 || throw(ArgumentError("`k` needs to be greater than 0"))
        new(k, Set{Variant}([]))
    end
end

function (ss::LibrarySelect)(variants::AbstractVector{Variant})
    map(variant -> push!(ss.library, variant), variants)
    selection = _select_top_k!(ss)
    #println(length(ss.library))
    return selection
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
