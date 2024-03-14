using PyCall

#@pyinclude(joinpath(@__DIR__, "..", "llm", "lib", "model", "single_layer", "TwoLayer.py"))
#model = py"""TwoLayer(activation_function=torch.nn.Sigmoid(), embedding_size=1280)"""
#println(model)

println(pyimport("sys")."path")
fp = pyimport("fitness_predictor")
a = fp.train_predictor
println(a)

importlib = pyimport("importlib")
yourmodule = pyimport("yourmodule")
importlib.reload(yourmodule)
