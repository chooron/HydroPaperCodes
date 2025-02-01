#* 运行ExpHydroV2, 加入下渗时的划分机制

using Pkg
Pkg.activate(".")

using DifferentialEquations
using Statistics
using ComponentArrays
using Interpolations
using ProgressMeter
using Optimization
using JLD2
using DelimitedFiles
using SciMLSensitivity
using CSV, DataFrames, Dates

include("models/exphydro.jl")
include("utils/train.jl")
include("utils/HydroNODE_data.jl")

input_var_names = ["Daylight(h)", "Prec(mm/day)", "Tmean(C)"]
output_var_name = "Flow(mm/s)"
data_path = "G:\\Dataset\\CAMELS_US"
model_name = "exphydroV2(disc,withst)" # 离散模型,考虑初始状态

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

	exphydro_model = build_exphydrov2([itp_Lday, itp_P, itp_T], solve_type = "discrete")
	lower_bounds = [0.0, 100.0, 10.0, 0.01, 0.0, -3.0, 0.01, 0.01, 100.0]
	upper_bounds = [0.1, 1500.0, 50.0, 5.0, 3.0, 0.0, 0.99, 1500.0, 1500.0]
	default_ps_vec = lower_bounds .+ 0.5 * (upper_bounds.-lower_bounds)
	loss_func = (y, y_hat) -> sum((y .- y_hat) .^ 2) / sum((y .- mean(y)) .^ 2)
	opt_params, opt_recorder = calibrate_exphydro_withst(
		exphydro_model, (train_arr, train_y, train_timepoints), default_ps_vec,
		lb = lower_bounds, ub = upper_bounds, max_N_iter = 10000, warmup = 1, # no warmup
		loss_func = loss_func
	)
	output = exphydro_model(total_arr, opt_params[1:7], opt_params[8:9], data_timepoints, return_all = true)
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
# basins_available = lpad.(string.(Int.(basins_id_file)), 8, "0")
basins_available = ["02315500", "03281500", "06191500"]
for basin_id in basins_available
	main(basin_id)
end

