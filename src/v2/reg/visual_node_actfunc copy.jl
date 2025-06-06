#* 将所有流域的数据打印起来然后绘制图片
using Lux
using JLD2
using Statistics
using Symbolics
using SymbolicUtils
using LaTeXStrings
using Random
using DataInterpolations

include("../models/m50.jl")
include("../../utils/kan_tools.jl")
include("../../utils/symbolic_relate.jl")
include("../utils/data_relate.jl")


function plot_activation_functions!(p, acts, postacts)
    for i in axes(postacts, 2)
        # 对每个输入维度进行循环
        for j in axes(acts, 1)
            current_acts = acts[j, :]
            # Sort acts and get indices
            sorted_indices = sortperm(current_acts)
            current_postacts = postacts[:, i, j]
            # 计算当前子图的索引
            plot_idx = j + (i - 1) * size(acts, 1) # 原来按行优先排序
            # 在对应的子图中绘制线条
            plot!(
                p[plot_idx],
                current_acts[sorted_indices], current_postacts[sorted_indices],
                # margin=1Plots.mm,
                # right_margin=4Plots.mm, bottom_margin=3Plots.mm,
                color=:grey,
                legend=false,
                linewidth=1,
                alpha=0.3,
            )
        end
    end
    return p
end

function load_data(basin_id, reg_gamma="1e-2")
    pruned_kan_ckpt = load("result/v2/k50_prune($(reg_gamma))/$(basin_id)/train_records.jld2")["reg_opt_ps"]
    other_info = load("result/v2/k50_prune($(reg_gamma))/$(basin_id)/other_info.jld2")
    output_df = other_info["output_df"]
    qnn_input = output_df[!, [:norm_snw, :norm_slw, :norm_infil]] |> Array |> permutedims
    qnn_nodes_to_keep = other_info["qnn_nodes_to_keep"]
    pruned_q_nn = Chain(
        KDense(3, length(qnn_nodes_to_keep), 6; use_base_act=true),
        KDense(length(qnn_nodes_to_keep), 1, 6; use_base_act=true),
        name=:qnn,
    )
    q_nn_params, q_nn_states = Lux.setup(Random.default_rng(), pruned_q_nn)
    qnn_params_axes = getaxes(q_nn_params |> ComponentArray)
    q_pas = ComponentVector(pruned_kan_ckpt["nns"]["qnn"], qnn_params_axes)
    qnn_layer1_postacts = activation_getter(pruned_q_nn.layer_1, q_pas.layer_1, q_nn_states.layer_1, qnn_input)
    qnn_layer1_postacts_arr = reduce((m1, m2) -> cat(m1, m2, dims=3), qnn_layer1_postacts)
    max_postacts1 = vec(maximum(maximum(abs.(qnn_layer1_postacts_arr), dims=1)[1, :, :], dims=2))
    sort_indices = sortperm(max_postacts1, rev=true)
    sort_qnn_layer1_postacts_arr = qnn_layer1_postacts_arr[:, sort_indices, :]
    return qnn_input, sort_qnn_layer1_postacts_arr
end

#* create a empty figure
fig = plot(
    layout=(3, 3),
    size=(600, 600),
    dpi=300,
    framestyle=:box,
    grid=false,
    margin=1Plots.mm,
    fontfamily="Times",
    wspace=0.1,
    hspace=0.1,
)

reg_gamma = "1e-2"
model_name = "k50_prune($(reg_gamma))"
base_path = "result/v2/$(model_name)"
basin_ids = filter(x -> isdir(joinpath(base_path, x)), readdir(base_path))

# 收集所有流域的数据
all_inputs, all_postacts = [], []
for basin_id in basin_ids
    qnn_input, layer1_postacts_arr = load_data(basin_id, reg_gamma)
    if size(layer1_postacts_arr, 2) <= 3
        plot_activation_functions!(fig, qnn_input[:, 365:end], layer1_postacts_arr[365:end, :, :])
        push!(all_postacts, layer1_postacts_arr[365:end, :, :])
        push!(all_inputs, qnn_input[:, 365:end])
    end
end

all_inputs_arr = stack(all_inputs, dims=1)
min_bounds = quantile.(eachslice(minimum(all_inputs_arr, dims=3)[:, :, 1], dims=2), Ref(0.25))
max_bounds = quantile.(eachslice(maximum(all_inputs_arr, dims=3)[:, :, 1], dims=2), Ref(0.75))

max_postacts_arr = stack(map(all_postacts) do postacts
    tmp_arr = zeros(3, 3)
    tmp_arr[1:size(postacts, 2), :] .= maximum(postacts, dims=1)[1, :, :]
    tmp_arr
end, dims=1)

min_postacts_arr = stack(map(all_postacts) do postacts
    tmp_arr = zeros(3, 3)
    tmp_arr[1:size(postacts, 2), :] .= minimum(postacts, dims=1)[1, :, :]
    tmp_arr
end, dims=1)

min_postact_bounds = reshape(floor.(quantile.(eachslice(min_postacts_arr, dims=(2, 3)), Ref(0.01)), digits=0)', 9)
max_postact_bounds = reshape(ceil.(quantile.(eachslice(max_postacts_arr, dims=(2, 3)), Ref(0.99)), digits=0)', 9)


itp_result_list = []
for i in 1:length(all_inputs)
    itp_list = map(eachslice(all_inputs[i], dims=1), eachslice(all_postacts[i], dims=3)) do input, postacts
        input_sorted_ind = sortperm(input)
        input_sorted = input[input_sorted_ind]
        postacts_sorted = postacts[input_sorted_ind, :]
        LinearInterpolation.(eachslice(postacts_sorted, dims=2), Ref(input_sorted); extrapolation=ExtrapolationType.Constant)
    end
    itp_result = map(zip(itp_list, eachslice(all_inputs[i], dims=1), min_bounds, max_bounds)) do (itps, input, min_bound, max_bound)
        zeros_output = zeros(3, length(min_bound:0.01:max_bound))
        output = stack([itp.(min_bound:0.01:max_bound) for itp in itps], dims=1)
        zeros_output[1:size(output, 1), :] .= output
        return zeros_output
    end
    push!(itp_result_list, itp_result)
end
snowpack_itp_result = mapslices(x -> median(filter(!iszero, x)), stack(map(i -> i[1], itp_result_list), dims=1), dims=1)[1, :, :]
soilwater_itp_result = mapslices(x -> median(filter(!iszero, x)), stack(map(i -> i[2], itp_result_list), dims=1), dims=1)[1, :, :]
rainfall_itp_result = mapslices(x -> median(filter(!iszero, x)), stack(map(i -> i[3], itp_result_list), dims=1), dims=1)[1, :, :]

colors = Colors.JULIA_LOGO_COLORS
for i in 1:3
    plot!(fig[(i-1)*3+1], min_bounds[1]:0.01:max_bounds[1], snowpack_itp_result[i, :],
        margin=2Plots.mm, right_margin=4Plots.mm, bottom_margin=3Plots.mm,
        color=colors[1], legend=false, linewidth=2)
    plot!(fig[(i-1)*3+2], min_bounds[2]:0.01:max_bounds[2], soilwater_itp_result[i, :],
        margin=2Plots.mm, right_margin=4Plots.mm, bottom_margin=3Plots.mm,
        color=colors[2], legend=false, linewidth=2)
    plot!(fig[(i-1)*3+3], min_bounds[3]:0.01:max_bounds[3], rainfall_itp_result[i, :],
        margin=2Plots.mm, right_margin=4Plots.mm, bottom_margin=3Plots.mm,
        color=colors[3], legend=false, linewidth=2)
end

for i in 1:9
    xaxis!(fig[i],
        showaxis=:both,        # 显示上下轴
        mirror=false,           # 开启镜像
        grid=true,
        tickfont=font(14, "Times"),
        linewidth=2,           # 增加轴线粗细
        xlims=(min_bounds[i % 3 == 0 ? 3 : i % 3], max_bounds[i % 3 == 0 ? 3 : i % 3]),
        xticks=round.(Int, range(min_bounds[i % 3 == 0 ? 3 : i % 3], max_bounds[i % 3 == 0 ? 3 : i % 3], length=4))
    )

    yaxis!(fig[i],
        showaxis=:both,        # 显示左右轴
        mirror=false,           # 开启镜像
        grid=true,
        tickfont=font(14, "Times"),
        linewidth=2,           # 增加轴线粗细
        ylims=(min_postact_bounds[i], max_postact_bounds[i]),
        yticks=round.(Int, range(min_postact_bounds[i], max_postact_bounds[i], length=4))
    )
end
savefig(fig, "src/v2/plots/figures/$(model_name)_avg_node_actfunc.png")