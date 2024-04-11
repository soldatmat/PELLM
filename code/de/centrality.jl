using FileIO

neighborhoods = load(joinpath(@__DIR__, "data", "neighborhoods", neighborhoods_filename))["neighborhoods"]

centrality = vec(zeros(size(neighborhoods)[2], 1))
foreach(x -> centrality[x] += 1, neighborhoods)

using Plots
histogram(centrality)
