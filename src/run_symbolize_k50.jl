using Lux
using Plots
using JLD2
using DifferentialEquations
using SciMLSensitivity
using StableRNGs
using Zygote
using Statistics
using ComponentArrays
using Interpolations
using ProgressMeter
using ParameterSchedulers: Scheduler, Exp, Step
using Optimization, OptimizationOptimisers
using CSV, DataFrames, Dates, DelimitedFiles

include("models/model_f.jl")
include("utils/train.jl")

basin_id = "06191500"
model_name = "k50_base(symbolize)"
include("postprocess/basin_$(basin_id)_q_kan_symbol.jl")
#* load data
camelsus_cache = load("src/data/camelsus/$(basin_id).jld2")
data_x, data_y, data_timepoints = camelsus_cache["data_x"], camelsus_cache["data_y"], camelsus_cache["data_timepoints"]
train_x, train_y, train_timepoints = camelsus_cache["train_x"], camelsus_cache["train_y"], camelsus_cache["train_timepoints"]
test_x, test_y, test_timepoints = camelsus_cache["test_x"], camelsus_cache["test_y"], camelsus_cache["test_timepoints"]
lday_vec, prcp_vec, temp_vec = collect(data_x[:, 1]), collect(data_x[:, 2]), collect(data_x[:, 3])
# Replace missing values with 0.0 in data arrays
for data_arr in [data_x, data_y, train_x, train_y, test_x, test_y]
	replace!(data_arr, missing => 0.0)
end

#* load exphydro model outputs
exphydro_df = CSV.read("src/result/models/exphydro(disc,withst)/$(basin_id)/model_outputs.csv", DataFrame)
snowpack_vec, soilwater_vec = exphydro_df[!, "snowpack"], exphydro_df[!, "soilwater"]
et_vec, qsim_vec, rainfall_vec, melt_vec = exphydro_df[!, "et"], exphydro_df[!, "qsim"], exphydro_df[!, "pr"], exphydro_df[!, "melt"]
et_vec[et_vec.<0] .= 0.000000001
qsim_vec[qsim_vec.<0] .= 0.000000001
rainfall_vec[rainfall_vec.<0] .= 0.000000001
melt_vec[melt_vec.<0] .= 0.000000001
infil_vec = rainfall_vec .+ melt_vec

#* define interpolation functions
itp_method = SteffenMonotonicInterpolation()
itp_Lday = interpolate(data_timepoints, lday_vec, itp_method)
itp_P = interpolate(data_timepoints, prcp_vec, itp_method)
itp_T = interpolate(data_timepoints, temp_vec, itp_method)
itpfuncs = (itp_P, itp_T, itp_Lday)

#* normalize data
s0_mean, s0_std = mean(snowpack_vec), std(snowpack_vec)
s1_mean, s1_std = mean(soilwater_vec), std(soilwater_vec)
temp_mean, temp_std = mean(temp_vec), std(temp_vec)
infil_mean, infil_std = mean(infil_vec), std(infil_vec)

norm_s0_func = (x) -> (x .- s0_mean) ./ s0_std
norm_s1_func = (x) -> (x .- s1_mean) ./ s1_std
norm_temp_func = (x) -> (x .- temp_mean) ./ temp_std
norm_infil_func = (x) -> (x .- infil_mean) ./ infil_std
normfuncs = (norm_s0_func, norm_s1_func, norm_infil_func)

norm_snowpack = norm_s0_func.(snowpack_vec)
norm_soilwater = norm_s1_func.(soilwater_vec)
norm_infil = norm_infil_func.(infil_vec)

#* prepare training data
qnn_train_input = permutedims(reduce(hcat, (norm_snowpack, norm_soilwater, norm_infil)))[:,1:length(train_timepoints)]
qnn_test_input = permutedims(reduce(hcat, (norm_snowpack, norm_soilwater, norm_infil)))[:,length(train_timepoints)+1:end]
symbolic_q_kan_train = symbolic_q_kan.(eachslice(qnn_train_input, dims=2), Ref(q_params))
opt_qsym_params, recorder_df = train_symbolic_func(
	symbolic_q_kan, (qnn_train_input, train_y),
	 q_params, loss_func=nse_loss, 
	 optmzr = Scheduler(ADAM, Step(start = 1e-2, step_sizes = [200, 200, 200, 200], decay = 0.5))
)

train_pred = symbolic_q_kan.(eachslice(qnn_train_input, dims=2), Ref(opt_qsym_params))
test_pred = symbolic_q_kan.(eachslice(qnn_test_input, dims=2), Ref(opt_qsym_params))
pretrain_train_nse = nse_loss(train_pred, train_y)
pretrain_test_nse = nse_loss(test_pred, test_y)

@info "pretrain_train_nse: $(pretrain_train_nse), pretrain_test_nse: $(pretrain_test_nse)"

#* load exphydro model parameters
exphydro_pas = load("src/result/models/exphydro(disc,withst)/$(basin_id)/opt_params.jld2")["opt_params"]
# Smax, Qmax, exphydro_initstates = exphydro_pas[2], exphydro_pas[3], exphydro_pas[end-1:end]
Smax, exphydro_initstates = exphydro_pas[2], exphydro_pas[end-1:end]
reopt_exphydro_params = load("src/result/models/k50_base(sparse)/$(basin_id)/train_records.jld2")["opt_ps"]["exphydro"]

model_init_pas = ComponentArray(q = opt_qsym_params, exphydro = vcat([Smax], reopt_exphydro_params)) # , et = et_params
train_arr = permutedims(train_x)[[2, 3, 1], :]
test_arr = permutedims(test_x)[[2, 3, 1], :]
total_arr = permutedims(data_x)[[2, 3, 1], :]

model_func = build_Symbolize_F(itpfuncs, normfuncs, symbolic_q_kan, initstates = exphydro_initstates, solve_type = "discrete")

# * define loss functions
opt_ps, val_opt_ps, loss_recorder = train_with_valv2(
	model_func, (train_arr, train_y, train_timepoints), (total_arr, data_y, data_timepoints), model_init_pas,
	loss_func = nse_loss, max_N_iter = 100, warm_up = 1, adtype=Optimization.AutoForwardDiff(), 
	optmzr = Scheduler(ADAM, Step(start = 1e-2, step_sizes = [30, 30], decay = 0.2)),
)

# * save calibration results
model_dir = "src/result/models/k50_base(symbolize)/$(basin_id)"
mkpath(model_dir)
save("$(model_dir)/train_records.jld2", "opt_ps", opt_ps, "val_opt_ps", val_opt_ps)
CSV.write("$(model_dir)/loss_df.csv", loss_recorder)

#* make prediction
y_total_pred = model_func(total_arr, opt_ps, data_timepoints)
y_val_total_pred = model_func(total_arr, val_opt_ps, data_timepoints)
y_train_pred = y_total_pred[1:length(train_timepoints)]
y_test_pred = y_total_pred[length(train_timepoints)+1:end]
y_val_train_pred = y_val_total_pred[1:length(train_timepoints)]
y_val_test_pred = y_val_total_pred[length(train_timepoints)+1:end]
train_predicted_df = DataFrame((pred = y_train_pred, val_pred = y_val_train_pred, obs = train_y))
test_predicted_df = DataFrame((pred = y_test_pred, val_pred = y_val_test_pred, obs = test_y))
CSV.write("$(model_dir)/train_predicted_df.csv", train_predicted_df)
CSV.write("$(model_dir)/test_predicted_df.csv", test_predicted_df)
