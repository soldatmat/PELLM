using Pkg

Pkg.activate(".")

##### Set the following variables according to your system. #####
OS = "macos" # Set to "windows", "linux" or "macos" depending on your OS.
CONDA_PATH = "/opt/miniconda3/" # Set path to Conda root folder here.


if OS == "windows"
    ENV["PYTHON"] = joinpath(CONDA_PATH, "envs", "pellm", "python.exe")
elseif OS == "linux" || OS == "macos"
    ENV["PYTHON"] = joinpath(CONDA_PATH, "envs", "pellm", "bin", "python")
end

Pkg.build("PyCall")

using PyCall
@show PyCall.python
print("PyCall setup complete. You need to restart Julia for PyCall to start using the specified python isntallation.")
