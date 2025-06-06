# 需要绘制每个激活函数的图形(未使用sparse的)
using JLD2
using CSV, DataFrames
using Plots
using Statistics
using Lux
using StableRNGs
using ComponentArrays
include("../models/model_f.jl")
include("../utils/data_relate.jl")
include("../utils/kan_tools.jl")

basin_id = "03281500"
model_name = "k50_base(reg)"
submodel_name = "exphydro_base(disc)"
model_dir = "src/result/models/$(model_name)/$(basin_id)"
kan_ckpt = load("$(model_dir)/train_records.jld2")["opt_ps"]
q_pas = kan_ckpt["q"]

q_nn_dims = size(q_pas.layer_1.C)[1]
q_nn = build_Q_NN(q_nn_dims, 6)
function plot_acts_func(acts, postacts)
    input_dims = size(acts, 1)
    activ_dims = size(postacts, 2)

    # 对每个输入维度进行循环
    for i in axes(acts, 1)
        current_acts = acts[i, :]
        # Sort acts and get indices
        sorted_indices = sortperm(current_acts)
        
        # 对每个激活函数进行循环
        for j in axes(postacts, 2)
            current_postacts = postacts[:, j, i]
            
            # 计算合适的坐标轴范围
            x_min, x_max = extrema(current_acts)
            y_min, y_max = extrema(current_postacts)
            
            # 计算x和y的范围
            x_range_size = x_max - x_min
            y_range_size = y_max - y_min
            
            # 使用较大的范围来确保比例一致
            max_range = max(x_range_size, y_range_size)
            
            # 重新计算x轴范围
            x_center = (x_max + x_min) / 2
            x_range = (x_center - max_range/2 , x_center + max_range/2)
            
            # 重新计算y轴范围
            y_center = (y_max + y_min) / 2
            y_range = (y_center - max_range/2, y_center + max_range/2)

            p = plot(current_acts[sorted_indices], current_postacts[sorted_indices],
                linewidth = 15,
                color = :black,
                aspect_ratio = 1,
                framestyle = :none,  # 移除边框
                grid = false,        # 移除网格
                ticks = false,       # 移除刻度
                size = (300, 300),   # 保持正方形
                xlims = x_range,
                ylims = y_range,
                margin = 0Plots.mm,  # 移除边距
                background = :transparent)  # 设置透明背景
            plot!(legend = false)
            savefig(p, "src/paper/figures/figure_1/layer2_act_$(i)_$(j).png")
        end
    end
end


#* load data
q_input = load_nn_data(basin_id, "exphydro(disc,withst)", "q")

q_nn_states = LuxCore.initialstates(StableRNG(1234), q_nn)
qnn_layer1_postacts = activation_getter(q_nn.layer_1, q_pas.layer_1, q_nn_states.layer_1, q_input)
qnn_layer1_postacts = reduce((m1, m2) -> cat(m1, m2, dims = 3), qnn_layer1_postacts)

qnn_layer1_output = q_nn.layer_1(q_input, q_pas.layer_1, q_nn_states.layer_1)[1]
qnn_layer2_postacts = activation_getter(q_nn.layer_2, q_pas.layer_2, q_nn_states.layer_2, qnn_layer1_output)
qnn_layer2_postacts = reduce((m1, m2) -> cat(m1, m2, dims = 3), qnn_layer2_postacts)

# plot_acts_func(q_input, qnn_layer1_postacts)
plot_acts_func(qnn_layer1_output, qnn_layer2_postacts)
