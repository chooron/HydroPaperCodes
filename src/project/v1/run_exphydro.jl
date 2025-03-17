using Pkg
Pkg.activate(".")

using Lux
using DifferentialEquations
using Statistics
using ComponentArrays
using Interpolations
using ProgressMeter
using Optimization
using Plots
using JLD2
using DelimitedFiles
using SciMLSensitivity
using CSV, DataFrames, Dates

include("../../models/exphydro.jl")
include("../../utils/train.jl")
include("../../utils/HydroNODE_data.jl")

input_var_names = ["Daylight(h)", "Prec(mm/day)", "Tmean(C)"]
output_var_name = "Flow(mm/s)"
data_path = "G:\\Dataset\\CAMELS_US"
model_name = "exphydro_1e-3"

function main(basin_id)
	camelsus_cache = load("src/data/camelsus/$(basin_id).jld2")
	data_x, data_y, data_timepoints = camelsus_cache["data_x"], camelsus_cache["data_y"], camelsus_cache["data_timepoints"]
	train_x, train_y, train_timepoints = camelsus_cache["train_x"], camelsus_cache["train_y"], camelsus_cache["train_timepoints"]
	test_x, test_y, test_timepoints = camelsus_cache["test_x"], camelsus_cache["test_y"], camelsus_cache["test_timepoints"]

	train_arr = permutedims(train_x)[[2, 3, 1], :]
	test_arr = permutedims(test_x)[[2, 3, 1], :]
	total_arr = permutedims(data_x)[[2, 3, 1], :]

	itp_method = SteffenMonotonicInterpolation()
	itp_Lday = interpolate(data_timepoints, data_x[:,1], itp_method)
	itp_P = interpolate(data_timepoints, data_x[:,2], itp_method)
	itp_T = interpolate(data_timepoints, data_x[:,3], itp_method)

	exphydro_model = build_exphydro([itp_Lday, itp_P, itp_T])
	lower_bounds = [0.0, 100.0, 10.0, 0.01, 0.0, -3.0]
	upper_bounds = [0.1, 2000.0, 50.0, 5.0, 3.0, 0.0]
	default_ps = @. (lower_bounds + upper_bounds) / 2

	initstates = [0.001, 0.001]
	loss_func = (y, y_hat) -> sum((y .- y_hat) .^ 2) / sum((y .- mean(y)) .^ 2)
	opt_params, opt_recorder = calibrate_hydromodel(
		exphydro_model, (train_arr, train_y, train_timepoints), default_ps, initstates,
		lb = lower_bounds, ub = upper_bounds, max_N_iter = 1000, warmup = 365,
		loss_func = loss_func
	)
	output = exphydro_model(total_arr, opt_params, initstates, data_timepoints, return_all = true)
	output_df = DataFrame(output)
	save_path = "src/result/$(model_name)/$(basin_id)"
	mkpath(save_path)
	CSV.write("$(save_path)/model_outputs.csv", output_df)
	CSV.write("$(save_path)/opt_recorder.csv", opt_recorder)
	train_pred_df = DataFrame((pred=output.qsim[1:length(train_timepoints)], obs=train_y))
	test_pred_df = DataFrame((pred=output.qsim[length(train_timepoints)+1:end], obs=test_y))
	CSV.write("$(save_path)/train_predicted_df.csv", train_pred_df)
	CSV.write("$(save_path)/test_predicted_df.csv", test_pred_df)
	save("$(save_path)/opt_params.jld2", "opt_params", opt_params)
end

basins_id_file = readdlm("src/data/basin_ids/basins_all.txt")
basins_available = lpad.(string.(Int.(basins_id_file)), 8, "0")
error_basins = []
for basin_id in basins_available
	try
		println("Processing basin $(basin_id)...")
		main(basin_id)
	catch e
		push!(error_basins, basin_id)
		println("Error processing basin $(basin_id): $(e)")
	end
end

