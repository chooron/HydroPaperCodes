# ! finetune kan(q) with different hidden dimensions and grid sizes
using Pkg
Pkg.activate(".")
using Lux
using Plots
using StableRNGs
using Zygote
using Statistics
using ComponentArrays
using HydroModels
using KolmogorovArnold
using Optimization, OptimizationOptimisers
using CSV, DataFrames, Dates, JLD2, JSON, DelimitedFiles
using CUDA, cuDNN
using Lux, LuxCUDA
CUDA.allowscalar(false)
device = Lux.gpu_device()
cpu_device = Lux.cpu_device()

include("../../utils/train.jl")
include("../../utils/criteria.jl")

function finetune_kan(basin_id, hidd_dims, grid_size)
    model_name = "kan(q)"
    basemodel_name = "exphydro(516)"
    basemodel_dir = "result/v2/$basemodel_name"
    #* load data
    camelsus_cache = load("data/camelsus/$(basin_id).jld2")
    data_x, data_y, data_timepoints = camelsus_cache["data_x"], camelsus_cache["data_y"], camelsus_cache["data_timepoints"]
    train_x, train_y, train_timepoints = camelsus_cache["train_x"], camelsus_cache["train_y"], camelsus_cache["train_timepoints"]
    test_x, test_y, test_timepoints = camelsus_cache["test_x"], camelsus_cache["test_y"], camelsus_cache["test_timepoints"]
    lday_vec, prcp_vec, temp_vec = collect(data_x[:, 1]), collect(data_x[:, 2]), collect(data_x[:, 3])
    # Replace missing values with 0.0 in data arrays
    for data_arr in [data_x, data_y, train_x, train_y, test_x, test_y]
        replace!(data_arr, missing => 0.0)
    end

    #* load exphydro model outputs
    exphydro_df = CSV.read("$(basemodel_dir)/$(basin_id)/model_outputs.csv", DataFrame)
    snowpack_vec, soilwater_vec = exphydro_df[!, "snowpack"], exphydro_df[!, "soilwater"]
    et_vec, flow_vec = exphydro_df[!, "et"], exphydro_df[!, "qsim"]
    pr_vec, melt_vec = exphydro_df[!, "pr"], exphydro_df[!, "melt"]
    infil_vec = pr_vec .+ melt_vec
    et_vec[et_vec.<0] .= 0.000000001
    flow_vec[flow_vec.<0] .= 0.000000001
    infil_vec[infil_vec.<0] .= 0.000000001

    #* normalize data
    s0_mean, s0_std = mean(snowpack_vec), std(snowpack_vec)
    s1_mean, s1_std = mean(soilwater_vec), std(soilwater_vec)
    infil_mean, infil_std = mean(infil_vec), std(infil_vec)

    norm_snowpack = (snowpack_vec .- s0_mean) ./ s0_std
    norm_soilwater = (soilwater_vec .- s1_mean) ./ s1_std
    norm_infil = (infil_vec .- infil_mean) ./ infil_std

    function build_kan(hidd_dims, grid_size)
        Chain(
            KDense(3, hidd_dims, grid_size; use_base_act=true),
            KDense(hidd_dims, 1, grid_size; use_base_act=true),
            name=:qnn,
        )
    end

    #* build model
    qnn = build_kan(hidd_dims, grid_size)
    q_nn_ps, q_nn_st = Lux.setup(StableRNG(42), qnn)
    q_nn_ps = ComponentVector(q_nn_ps) .|> Float64 |> device
    q_nn_st = device(q_nn_st)
    q_apply = (x, p) -> qnn(x, p, q_nn_st)[1]
    #* prepare training data
    qnn_input = stack([norm_snowpack, norm_soilwater, norm_infil], dims=1) |> device
    qnn_target = reshape(flow_vec, 1, :) |> device
    #* train NNs
    opt_q_ps, qnn_loss_recorder = train_nn_v2(
        q_apply, (qnn_input[:, 1:length(train_timepoints)], qnn_target[:, 1:length(train_timepoints)]),
        (qnn_input[:, length(train_timepoints)+1:end], qnn_target[:, length(train_timepoints)+1:end]),
        q_nn_ps, optmzr=ADAM(0.01), max_N_iter=2000,
        # reg_func=reg_loss(1e-3, [:layer_1, :layer_2])
    )
    train_pred = q_apply(qnn_input[:, 1:length(train_timepoints)], opt_q_ps)
    test_pred = q_apply(qnn_input[:, length(train_timepoints)+1:end], opt_q_ps)
    train_mse = mse_loss(train_pred[1, :], flow_vec[1:length(train_timepoints)] |> device)
    test_mse = mse_loss(test_pred[1, :], flow_vec[length(train_timepoints)+1:end] |> device)
    mkpath("result/finetune/$(model_name)_$(hidd_dims)_$(grid_size)")
    open("result/finetune/$(model_name)_$(hidd_dims)_$(grid_size)/$(basin_id).json", "w") do f
        JSON.print(f, Dict("train_mse" => train_mse, "test_mse" => test_mse))
    end
end

basin_file = readdlm(joinpath("data/basin_ids/basins_all.txt"), ',')
basins_available = lpad.(string.(Int.(basin_file[:, 1])), 8, "0")
# grid_size = 6
# for basin_id in basins_available
#     for hidd_dims in [4, 5, 6, 7, 8]
#         finetune_kan(basin_id, hidd_dims, grid_size)
#     end
# end

hidd_dims = 6
for basin_id in basins_available
    for grid_size in [4, 5, 7, 8]
        finetune_kan(basin_id, hidd_dims, grid_size)
    end
end