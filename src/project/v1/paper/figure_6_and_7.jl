# 拟合KAN的激活曲线然后构建符号公式


using Lux
using JLD2
using SymbolicRegression
using Symbolics
using SymbolicUtils
using LaTeXStrings

include("../utils/kan_tools.jl")
include("../models/model_f.jl")
include("../utils/symbolic_relate.jl")
include("../utils/data_relate.jl")


function plot_activation_functions(acts, postacts; alpha_scale = 1.0, reverse =false)
	input_dims = size(acts, 1)
	activ_dims = size(postacts, 2)

	# 创建一个网格布局的图
    plot_size,layout = if reverse
        (300 * activ_dims, 250 * input_dims), (input_dims, activ_dims)
    else
        (300 * input_dims, 250 * activ_dims), (activ_dims, input_dims)
    end

	p = plot(layout = layout,
		size = plot_size,
		legend = false)

	# 对每个输入维度进行循环
	for i in axes(acts, 1)
		current_acts = acts[i, :]
		# Sort acts and get indices
		sorted_indices = sortperm(current_acts)
		acts_scale = maximum(current_acts) - minimum(current_acts)

		# 对每个激活函数进行循环
		for j in axes(postacts, 2)
			current_postacts = postacts[:, j, i]

			# 计算透明度
			alpha = tanh(alpha_scale * (maximum(current_postacts) - minimum(current_postacts)) / acts_scale)
			color = alpha < 1e-1 ? :grey : :blue
			alpha = alpha < 1e-1 ? 0.5 : alpha

			# 计算当前子图的索引
			plot_idx = j + (i-1) * activ_dims # 原来按行优先排序

			x_range = extrema(current_acts)
			y_range = extrema(current_postacts)
			xticks = range(x_range[1], x_range[2], length = 3)
			yticks = range(y_range[1], y_range[2], length = 3)

			# 在对应的子图中绘制线条
			plot!(p[plot_idx], current_acts[sorted_indices], current_postacts[sorted_indices],
				alpha = alpha,
				linewidth = 6,
				color = color,
				xticks = xticks,
				yticks = yticks,
				fontfamily = "Times",
				xformatter = x -> @sprintf("%.2f", x),  # 格式化为2位小数
				yformatter = y -> @sprintf("%.2f", y),
                guidefontsize=16,
                dpi=300,
                margin=2Plots.mm,
				bottom_margin=3Plots.mm,
                tickfont=font(14, "Times"),
                framestyle = :box,
                grid = false,
            )
			xaxis!(p[plot_idx], 
				showaxis=:both,        # 显示上下轴
				mirror=false,           # 开启镜像
				tickfontsize=16,
				guidefontsize=16,
				grid=true,
				linewidth=2)           # 增加轴线粗细

			yaxis!(p[plot_idx], 
				showaxis=:both,        # 显示左右轴
				mirror=false,           # 开启镜像
				tickfontsize=16,
				guidefontsize=16,
				grid=true,
				linewidth=2)           # 增加轴线粗细
			# 在图内添加标注
			annotate!(p[plot_idx], x_range[1] + 0.1 * (x_range[2] - x_range[1]),
				y_range[2] - 0.1 * (y_range[2] - y_range[1]), text("𝛷(2,$(i),$(j))", 18, :left, "Times"))
		end
	end

	return p

end

# basin_id = "03281500"  # 02361000, 03281500, 06191500  # call by other file
# basin_id = "02361000"
for basin_id in ["02361000", "03281500", "06191500"]
	model_name = "k50_base(sparse)"
	submodel_name = "exphydro_base(disc)"
	model_dir = "src/result/models/$(model_name)/$(basin_id)"
	kan_ckpt = load("$(model_dir)/train_records.jld2")["opt_ps"]
	q_pas, q_pas = kan_ckpt["et"], kan_ckpt["q"]

	q_nn_dims = size(q_pas.layer_1.C)[1]
	@info "pruned q_nn_dims: $q_nn_dims"
	q_nn = build_Q_NN(q_nn_dims, 6)
	#* load data

	q_input = load_nn_data(basin_id, "exphydro(disc,withst)", "q")

	q_nn_states = LuxCore.initialstates(StableRNG(1234), q_nn)
	qnn_layer1_postacts = activation_getter(q_nn.layer_1, q_pas.layer_1, q_nn_states.layer_1, q_input)
	qnn_layer1_postacts_arr = reduce((m1, m2) -> cat(m1, m2, dims = 3), qnn_layer1_postacts)

	qnn_layer1_output = q_nn.layer_1(q_input, q_pas.layer_1, q_nn_states.layer_1)[1]
	qnn_layer2_postacts = activation_getter(q_nn.layer_2, q_pas.layer_2, q_nn_states.layer_2, qnn_layer1_output)
	qnn_layer2_postacts_arr = reduce((m1, m2) -> cat(m1, m2, dims = 3), qnn_layer2_postacts)

	# p1 = plot_activation_functions(q_input, qnn_layer1_postacts_arr, reverse=true)
	p2 = plot_activation_functions(qnn_layer1_output, qnn_layer2_postacts_arr)

	mkpath("src/paper/figures/figure_6/$basin_id")
	# savefig(p1, "src/paper/figures/figure_6/$basin_id/q_nn_layer1_postacts.png")
	savefig(p2, "src/paper/figures/figure_6/$basin_id/q_nn_layer2_postacts.png")
end