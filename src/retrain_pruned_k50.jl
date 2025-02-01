#= 
* 使用正则化约束下的模型进行训练后,进行剪枝操作,并将剪枝的结果重新进行训练
=#
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
include("utils/kan_tools.jl")

etnn_prune_threshold = 0.1
qnn_prune_threshold = 1.2 # 03281500 (2)

basin_id = "03281500"
model_type = "base"
load_model_name = "k50_$model_type(reg)"
basemodel_name = "exphydro_$model_type(disc)"
save_model_name = "k50_$model_type(sparse)"

#* load parameters and initial states
exphydro_pas = load("src/result/models/$basemodel_name/$(basin_id)/opt_params.jld2")["opt_params"]
if basemodel_name == "exphydro_pro_2(disc)"
	exphydro_params, exphydro_init_states = exphydro_pas[4:6], exphydro_pas[end-1:end]
elseif basemodel_name == "exphydro_pro_1(disc)"
	exphydro_params, exphydro_init_states = exphydro_pas[3:5], exphydro_pas[end-1:end]
elseif basemodel_name == "exphydro_base(disc)"
	exphydro_params, exphydro_init_states = exphydro_pas[4:6], exphydro_pas[7:8]
end

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
exphydro_df = CSV.read("src/result/models/$basemodel_name/$(basin_id)/model_outputs.csv", DataFrame)
snowpack_vec, soilwater_vec = exphydro_df[!, "snowpack"], exphydro_df[!, "soilwater"]
et_vec, qsim_vec, rainfall_vec, melt_vec = exphydro_df[!, "et"], exphydro_df[!, "qsim"], exphydro_df[!, "pr"], exphydro_df[!, "melt"]
et_vec[et_vec.<0] .= 0.000000001
qsim_vec[qsim_vec.<0] .= 0.000000001
rainfall_vec[rainfall_vec.<0] .= 0.000000001
melt_vec[melt_vec.<0] .= 0.000000001
infil_vec = melt_vec .+ rainfall_vec

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

#* prepare training data
nn_input = permutedims(reduce(hcat, (norm_snowpack, norm_soilwater, norm_temp, norm_infil)))
et_nn_input = nn_input[[1, 2, 3], 1:length(train_timepoints)]
et_nn_target = reshape(et_vec[1:length(train_timepoints)] ./ lday_vec[1:length(train_timepoints)], 1, :)
q_nn_input = nn_input[[1, 2, 4], 1:length(train_timepoints)]
q_nn_target = reshape(qsim_vec[1:length(train_timepoints)], 1, :)
train_arr = permutedims(train_x)[[2, 3, 1], :]
test_arr = permutedims(test_x)[[2, 3, 1], :]
total_arr = permutedims(data_x)[[2, 3, 1], :]

#* load prune results
kan_ckpt = load("src/result/models/$load_model_name/$basin_id/train_records.jld2")["opt_ps"]
original_et_nn, original_q_nn = build_ET_NN(6, 6), build_Q_NN(6, 6)
pruned_et_nn_layers, pruned_et_ps = prune_nodes(
	model_layers = [original_et_nn.layer_1, original_et_nn.layer_2], layer_params = kan_ckpt["et"],
	input_data = et_nn_input, prune_threshold = etnn_prune_threshold,
)
pruned_et_nn = Lux.Chain(pruned_et_nn_layers...)
pruned_et_nn_st = LuxCore.initialstates(StableRNG(42), pruned_et_nn)

pruned_q_nn_layers, pruned_q_ps = prune_nodes(
	model_layers = [original_q_nn.layer_1, original_q_nn.layer_2], layer_params = kan_ckpt["q"],
	input_data = q_nn_input, prune_threshold = qnn_prune_threshold,
)
pruned_q_nn = Lux.Chain(pruned_q_nn_layers...)
pruned_q_nn_st = LuxCore.initialstates(StableRNG(42), pruned_q_nn)
@info "pruned et_nn_dims: $(size(pruned_et_ps.layer_1.C)[1]), pruned q_nn_dims: $(size(pruned_q_ps.layer_1.C)[1])"
et_apply = (x, p) -> pruned_et_nn(x, p, pruned_et_nn_st)[1]
q_apply = (x, p) -> pruned_q_nn(x, p, pruned_q_nn_st)[1]

model_func = build_Model_F(itpfuncs, normfuncs, (et_apply, q_apply), initstates = exphydro_init_states, solve_type = "discrete")
model_init_pas = ComponentArray(et = pruned_et_ps, q = pruned_q_ps, exphydro = exphydro_params)

# retrain model
opt_ps, val_opt_ps, loss_recorder = train_with_valv2(
	model_func, (train_arr, train_y, train_timepoints), (total_arr, data_y, data_timepoints), model_init_pas,
	loss_func = nse_loss, max_N_iter = 200, warm_up = 1,  reg_func = reg_loss(5e-3, [:et, :q]),
	optmzr = Scheduler(ADAM, Step(5e-3, 0.5, [50, 50, 50])),
)

# * save calibration results

model_dir = "src/result/models/$(save_model_name)/$(basin_id)"
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

