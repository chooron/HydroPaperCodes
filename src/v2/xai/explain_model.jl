# 使用ExplainableAI对模型的
using Lux
using Zygote
using JLD2, CSV, DataFrames, DelimitedFiles
using ExplainableAI
using StableRNGs
using Statistics
using Plots

include("../models/nns.jl")
include("../models/m50.jl")
include("../utils/data_relate.jl")

function main(basin_id)
    basemodel_name = "exphydro(516)"
    @info "Loading data for basin $basin_id"
    et_input, qnn_input = load_nn_data(basin_id, basemodel_name)
    etnn, qnn = build_nns()
    qnn_train_params = load("result/v2/k50_reg/$basin_id/train_records.jld2")["opt_ps"][:nns][:qnn]
    qnn_ps, qnn_st = Lux.setup(StableRNG(42), qnn)
    qnn_ps_axes = getaxes(ComponentArray(qnn_ps))
    qnn_func(x) = qnn(x, ComponentArray(qnn_train_params, qnn_ps_axes), qnn_st)[1]

    analyzer1 = IntegratedGradients(qnn_func)
    analyzer2 = SmoothGrad(qnn_func)
    expl1 = analyze(qnn_input, analyzer1)
    expl2 = analyze(qnn_input, analyzer2)

    exp1_df = DataFrame((s0=expl1.val[1, :], s1=expl1.val[2, :], infil=expl1.val[3, :], output=expl1.output[1, :]))
    exp2_df = DataFrame((s0=expl2.val[1, :], s1=expl2.val[2, :], infil=expl2.val[3, :], output=expl2.output[1, :]))

    expl1_mean = (basin_id=basin_id, S0=mean(abs, exp1_df.s0), S1=mean(abs, exp1_df.s1), I=mean(abs, exp1_df.infil))
    expl2_mean = (basin_id=basin_id, S0=mean(abs, exp2_df.s0), S1=mean(abs, exp2_df.s1), I=mean(abs, exp2_df.infil))
    return expl1_mean, expl2_mean
end

basin_file = readdlm(joinpath("data/basin_ids/basins_all.txt"), ',')
basins_available = lpad.(string.(Int.(basin_file[:, 1])), 8, "0")

ig_values_list = []
sg_values_list = []

for basin_id in basins_available
    ig_values, sg_values = main(basin_id)
    push!(ig_values_list, ig_values)
    push!(sg_values_list, sg_values)
end

camels_ig_values = DataFrame(ig_values_list)
camels_sg_values = DataFrame(sg_values_list)
CSV.write("result/xai/camels_ig_values_kan2.csv", camels_ig_values)
CSV.write("result/xai/camels_sg_values_kan2.csv", camels_sg_values)

