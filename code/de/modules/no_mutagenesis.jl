struct NoMutagenesis <: DESilico.Mutagenesis end
(::NoMutagenesis)(parents::AbstractVector{Vector{Char}}) = parents
