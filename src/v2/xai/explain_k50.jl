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

function main(basin_id, model_name)
    @info "Loading data for basin $basin_id"
    etnn, qnn = build_nns()
    qnn_train_params = load("result/v2/$(model_name)/$basin_id/train_records.jld2")["opt_ps"][:nns][:qnn]
    output_df = load("result/v2/$(model_name)/$basin_id/other_info.jld2")["output_df"]
    qnn_input = output_df[!, [:norm_snw, :norm_slw, :norm_infil]] |> Array |> permutedims
    qnn_ps, qnn_st = Lux.setup(StableRNG(42), qnn)
    qnn_ps_axes = getaxes(ComponentArray(qnn_ps))
    qnn_func(x) = qnn(x, ComponentArray(qnn_train_params, qnn_ps_axes), qnn_st)[1]

    analyzer1 = IntegratedGradients(qnn_func)
    analyzer2 = SmoothGrad(qnn_func)
    expl1 = analyze(qnn_input[:, 365:end], analyzer1)
    expl2 = analyze(qnn_input[:, 365:end], analyzer2)

    exp1_df = DataFrame((s0=expl1.val[1, :], s1=expl1.val[2, :], infil=expl1.val[3, :], output=expl1.output[1, :]))
    exp2_df = DataFrame((s0=expl2.val[1, :], s1=expl2.val[2, :], infil=expl2.val[3, :], output=expl2.output[1, :]))

    expl1_median = (basin_id=basin_id, S0=mean(abs, exp1_df.s0), S1=mean(abs, exp1_df.s1), I=mean(abs, exp1_df.infil))
    expl2_median = (basin_id=basin_id, S0=mean(abs, exp2_df.s0), S1=mean(abs, exp2_df.s1), I=mean(abs, exp2_df.infil))
    return expl1_median, expl2_median
end

basin_file = readdlm(joinpath("data/basin_ids/basins_all.txt"), ',')
basins_available = lpad.(string.(Int.(basin_file[:, 1])), 8, "0")

ig_values_list = []
sg_values_list = []
model_name = "k50_reg(1e-2)"

for basin_id in basins_available
    ig_values, sg_values = main(basin_id, model_name)
    push!(ig_values_list, ig_values)
    push!(sg_values_list, sg_values)
end

camels_ig_values = DataFrame(ig_values_list)
camels_sg_values = DataFrame(sg_values_list)
CSV.write("result/xai/camels_ig_values_$(model_name).csv", camels_ig_values)
CSV.write("result/xai/camels_sg_values_$(model_name).csv", camels_sg_values)

