using Pkg

Pkg.activate(".")

CONDA_PATH = "C:\\Users\\matou\\anaconda3" # Set path to Conda root folder here.
ENV["PYTHON"] = joinpath(CONDA_PATH, "envs", "pellm", "python.exe")
Pkg.build("PyCall")
