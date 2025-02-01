#* 运行符号化K50模型
using Pkg
Pkg.activate(".")

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

include("../models/model_f.jl")
include("../utils/train.jl")
include("basin_02361000_q_kan_symbol.jl")

basin_id = "02361000"
model_name = "k50(sparse)"
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
exphydro_df = CSV.read("src/result/exphydro(disc,withst)/$(basin_id)/model_outputs.csv", DataFrame)
snowpack_vec, soilwater_vec = exphydro_df[!, "snowpack"], exphydro_df[!, "soilwater"]
et_vec, qsim_vec, rainfall_vec = exphydro_df[!, "et"], exphydro_df[!, "qsim"], exphydro_df[!, "pr"]
et_vec[et_vec.<0] .= 0.000000001
qsim_vec[qsim_vec.<0] .= 0.000000001
rainfall_vec[rainfall_vec.<0] .= 0.000000001

#* define interpolation functions
itp_method = SteffenMonotonicInterpolation()
itp_Lday = interpolate(data_timepoints, lday_vec, itp_method)
itp_P = interpolate(data_timepoints, prcp_vec, itp_method)
itp_T = interpolate(data_timepoints, temp_vec, itp_method)
itpfuncs = (itp_P, itp_T, itp_Lday)

#* normalize data
s0_mean, s0_std = mean(snowpack_vec), std(snowpack_vec)
s1_mean, s1_std = mean(soilwater_vec), std(soilwater_vec)
lday_mean, lday_std = mean(lday_vec), std(lday_vec)
t_mean, t_std = mean(temp_vec), std(temp_vec)
rainfall_mean, rainfall_std = mean(rainfall_vec), std(rainfall_vec)

norm_s0_func = (x) -> (x .- s0_mean) ./ s0_std
norm_s1_func = (x) -> (x .- s1_mean) ./ s1_std
norm_lday_func = (x) -> (x .- lday_mean) ./ lday_std
norm_temp_func = (x) -> (x .- t_mean) ./ t_std
norm_rainfall_func = (x) -> (x .- rainfall_mean) ./ rainfall_std
normfuncs = (norm_s0_func, norm_s1_func, norm_lday_func, norm_temp_func, norm_rainfall_func)

norm_snowpack = norm_s0_func.(snowpack_vec)
norm_soilwater = norm_s1_func.(soilwater_vec)
norm_rainfall = norm_rainfall_func.(rainfall_vec)
norm_lday = norm_lday_func.(lday_vec)
norm_temp = norm_temp_func.(temp_vec)

#* load exphydro model parameters
exphydro_pas = load("src/result/exphydro(disc,withst)/$(basin_id)/opt_params.jld2")["opt_params"]
exphydro_initstates = exphydro_pas[7:8]
reopt_exphydro_params = load("src/result/$model_name/$(basin_id)/pruned_train_records.jld2")["opt_ps"]["exphydro"]
pruned_etnn_params = load("src/result/$model_name/$(basin_id)/pruned_train_records.jld2")["opt_ps"]["et"]
pruned_etnn = build_ET_NN(size(pruned_etnn_params.layer_1.C)[1], 6)
pruned_etnn_states = LuxCore.initialstates(StableRNG(42), pruned_etnn)
etnn_apply = (x, p) -> pruned_etnn(x, p, pruned_etnn_states)[1]
model_init_pas = ComponentArray(et = pruned_etnn_params, q = q_params, exphydro = reopt_exphydro_params)
train_arr = permutedims(train_x)[[2, 3, 1], :]
test_arr = permutedims(test_x)[[2, 3, 1], :]
total_arr = permutedims(data_x)[[2, 3, 1], :]
#* build hybrid model
model_func = build_Symbolize_F(itpfuncs, normfuncs, (etnn_apply, symbolic_q_kan), initstates = exphydro_initstates, solve_type = "discrete")
qout = model_func(train_arr, model_init_pas, train_timepoints)

#* define loss functions
loss_func(obs, pred) = sum((pred .- obs) .^ 2) / sum((obs .- mean(obs)) .^ 2)
opt_ps, val_opt_ps, loss_recorder = train_with_valv2(
	model_func, (train_arr, train_y, train_timepoints), (total_arr, data_y, data_timepoints), model_init_pas,
	loss_func = loss_func, max_N_iter = 200, warm_up = 1, adtype=Optimization.AutoForwardDiff(), reg_func = reg_loss(1e-3),
	optmzr = ADAM(0.01),
)
# q_pred = model_func(test_arr, opt_ps, test_timepoints)
# plot(vec(q_pred), label = "symbolize_q_kan")
# plot!(test_y, label = "exphydro")
