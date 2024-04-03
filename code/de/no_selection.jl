"""
TODO
"""
struct NoSelection <: DESilico.SelectionStrategy end

(::NoSelection)(variants::AbstractVector{Variant}) = map(variant -> variant.sequence, variants)
