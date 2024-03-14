function _mutate_sequence(sequence::Vector{Char}, mutation_positions::Vector{Int}, residues::AbstractVector{Vector{Char}})
    map(r -> _mutate_sequence(sequence, mutation_positions, r), residues)
end
function _mutate_sequence(sequence::Vector{Char}, mutation_positions::Vector{Int}, residues::Vector{Char})
    foreach(i -> sequence[mutation_positions[i]] = residues[i], 1:length(mutation_positions))
end
