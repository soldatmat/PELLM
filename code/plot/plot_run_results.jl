using Plots

# Histogram
histogram(map(v -> v.fitness, [v for v in sequence_space.variants]), bins=range(0.0, 1.01, length=20))

# Top fitness history
history = Vector{Variant}([sequence_space.variants[1]])
map(v -> push!(history, v.fitness > history[end].fitness ? v : history[end]), sequence_space.variants[2:end])
plot(0:length(history)-1, map(v -> v.fitness, history), ylim=(0.0, 1.0))

# Extract residues
extract_residues(sequence::Vector{Char}) = sequence[mutation_positions]
residues = map(v -> join(extract_residues(v.sequence)), sequence_space.variants)
residues_top = map(v -> join(extract_residues(v.sequence)), history)
