#= 
* 使用Discrete solve
* 蒸发模型输入为归一化后的snowpack, soilwater, temp, lday
* 径流模型输入为归一化后的snowpack, soilwater, rainfall
* 输出无其他转换例如log
 =#

using Lux
using Plots
using JLD2
using DifferentialEquations
using SciMLSensitivity
using StableRNGs
using Statistics
using ComponentArrays
using Interpolations
using ProgressMeter
using ParameterSchedulers: Scheduler, Exp, Step
using Optimization, OptimizationOptimisers
using CSV, DataFrames, Dates, DelimitedFiles

include("models/k50v2.jl")
include("../../utils/train.jl")


basemodel_name = "exphydro(disc,withst)"
nn_models_dict = Dict("k50_base(step)" => build_K50_NNs(6, 6))

basin_id = "01013500"
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

#* load parameters and initial states
exphydro_pas = load("src/result/models/$basemodel_name/$(basin_id)/opt_params.jld2")["opt_params"]

if basemodel_name == "exphydro_pro_2(disc)"
	exphydro_params, exphydro_init_states = exphydro_pas[4:6], exphydro_pas[end-1:end]
elseif basemodel_name == "exphydro_pro_1(disc)"
	exphydro_params, exphydro_init_states = exphydro_pas[3:5], exphydro_pas[end-1:end]
else
	exphydro_params, exphydro_init_states = exphydro_pas[4:6], exphydro_pas[7:8]
end

#* load exphydro model outputs
exphydro_df = CSV.read("src/result/models/$basemodel_name/$(basin_id)/model_outputs.csv", DataFrame)
snowpack_vec, soilwater_vec = exphydro_df[!, "snowpack"], exphydro_df[!, "soilwater"]
et_vec, qsim_vec, rainfall_vec, melting_vec = exphydro_df[!, "et"], exphydro_df[!, "qsim"], exphydro_df[!, "pr"], exphydro_df[!, "melt"]
et_vec[et_vec.<0] .= 0.000000001
qsim_vec[qsim_vec.<0] .= 0.000000001
rainfall_vec[rainfall_vec.<0] .= 0.000000001
melting_vec[melting_vec.<0] .= 0.000000001
infil_vec = rainfall_vec .+ melting_vec

#* define interpolation functions
itp_method = SteffenMonotonicInterpolation()
itp_Lday = interpolate(data_timepoints, lday_vec, itp_method)
itp_P = interpolate(data_timepoints, prcp_vec, itp_method)
itp_T = interpolate(data_timepoints, temp_vec, itp_method)
itpfuncs = (itp_P, itp_T, itp_Lday)

#* normalize data
s0_mean, s0_std = mean(snowpack_vec), std(snowpack_vec)
s1_mean, s1_std = mean(soilwater_vec), std(soilwater_vec)
t_mean, t_std = mean(temp_vec), std(temp_vec)
infil_mean, infil_std = mean(infil_vec), std(infil_vec)

norm_s0_func = (x) -> (x .- s0_mean) ./ s0_std
norm_s1_func = (x) -> (x .- s1_mean) ./ s1_std
norm_temp_func = (x) -> (x .- t_mean) ./ t_std
norm_infil_func = (x) -> (x .- infil_mean) ./ infil_std
normfuncs = (norm_s0_func, norm_s1_func, norm_temp_func, norm_infil_func)

norm_snowpack = norm_s0_func.(snowpack_vec)
norm_soilwater = norm_s1_func.(soilwater_vec)
norm_temp = norm_temp_func.(temp_vec)
norm_infil = norm_infil_func.(infil_vec)

#* build model
et_nn, q_nn = build_K50_NNs(6, 6)
et_nn_ps, et_nn_st = Lux.setup(StableRNG(42), et_nn)
q_nn_ps, q_nn_st = Lux.setup(StableRNG(42), q_nn)
et_apply = (x, p) -> et_nn(x, p, et_nn_st)[1]
q_apply = (x, p) -> q_nn(x, p, q_nn_st)[1]
model_func = build_Model_F(itpfuncs, normfuncs, (et_apply, q_apply), initstates = exphydro_init_states, solve_type = "discrete")
#* prepare training data
nn_input = permutedims(reduce(hcat, (norm_snowpack, norm_soilwater, norm_temp, norm_infil)))
et_ratio = et_vec[1:length(train_timepoints)] ./ lday_vec[1:length(train_timepoints)]
etnn_input = nn_input[[1, 2, 3], 1:length(train_timepoints)]
etnn_target = reshape(et_ratio, 1, :)

qnn_input = nn_input[[1, 2, 4], 1:length(train_timepoints)]
qnn_target = reshape(qsim_vec[1:length(train_timepoints)], 1, :)

#* train NNs
opt_et_ps, etnn_loss_recorder = train_NN(et_apply, (etnn_input, etnn_target), ComponentVector(et_nn_ps), optmzr = ADAM(0.01), max_N_iter = 2000)
opt_q_ps, qnn_loss_recorder = train_NN(q_apply, (qnn_input, qnn_target), ComponentVector(q_nn_ps), optmzr = ADAM(0.01), max_N_iter = 2000)

model_init_pas = ComponentArray(et = opt_et_ps, q = opt_q_ps, exphydro = exphydro_params)

train_arr = permutedims(train_x)[[2, 3, 1], :]
test_arr = permutedims(test_x)[[2, 3, 1], :]
total_arr = permutedims(data_x)[[2, 3, 1], :]

train_result = model_func(train_arr, model_init_pas, train_timepoints)
#* define loss functions
loss_func(obs, pred) = sum((pred .- obs) .^ 2) / sum((obs .- mean(obs)) .^ 2)

opt_ps, val_opt_ps, loss_recorder = train_with_val(
	model_func, (train_arr, train_y, train_timepoints), (total_arr, data_y, data_timepoints), model_init_pas,
	loss_func = loss_func, max_N_iter = 100, warm_up = 1, optmzr = ADAM(1e-2),
)

# # * save calibration results
# model_dir = "src/result/models/$(model_name)/$(basin_id)"
# mkpath(model_dir)
# save("$(model_dir)/train_records.jld2", "opt_ps", opt_ps, "val_opt_ps", val_opt_ps, "opt_en_ps", opt_et_ps, "opt_q_ps", opt_q_ps)
# CSV.write("$(model_dir)/etnn_loss_df.csv", etnn_loss_recorder)
# CSV.write("$(model_dir)/qnn_loss_df.csv", qnn_loss_recorder)
# CSV.write("$(model_dir)/loss_df.csv", loss_recorder)

# #* make prediction
# y_total_pred = model_func(total_arr, opt_ps, data_timepoints)
# y_val_total_pred = model_func(total_arr, val_opt_ps, data_timepoints)
# y_train_pred = y_total_pred[1:length(train_timepoints)]
# y_test_pred = y_total_pred[length(train_timepoints)+1:end]
# y_val_train_pred = y_val_total_pred[1:length(train_timepoints)]
# y_val_test_pred = y_val_total_pred[length(train_timepoints)+1:end]
# train_predicted_df = DataFrame((pred = y_train_pred, val_pred = y_val_train_pred, obs = train_y))
# test_predicted_df = DataFrame((pred = y_test_pred, val_pred = y_val_test_pred, obs = test_y))
# CSV.write("$(model_dir)/train_predicted_df.csv", train_predicted_df)
# CSV.write("$(model_dir)/test_predicted_df.csv", test_predicted_df)
# GC.gc(true)