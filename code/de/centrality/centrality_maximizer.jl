"""
TODO
"""
#= struct CentralityMaximizer <: DESilico.Mutagenesis
    centrality::Vector{Int}
end

function CentralityMaximizer(neighborhoods::Matrix{Int})
    centrality = vec(zeros(Int, (size(neighborhoods)[2], 1)))
    foreach(x -> centrality[x] += 1, neighborhoods)
    CentralityMaximizer(centrality)
end =#

function get_top_centrality(neighborhoods::Matrix{Int}, k::Int)
    centrality = vec(zeros(Int, (size(neighborhoods)[2], 1)))
    foreach(x -> centrality[x] += 1, neighborhoods)
    variants = collect(enumerate(centrality))
    sort!(variants, by=x -> x[2], rev=true)
    map(v -> v[1], variants[1:k])
end
