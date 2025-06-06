# 绘制Symbolize模型的计算过程.
using Symbolics
using ComponentArrays
using Plots
using Statistics
using Lux
using StableRNGs

include("../utils/symbolic_relate.jl")
include("../utils/kan_tools.jl")
include("../utils/data_relate.jl")
include("../models/model_f.jl")

model_name = "k50_base(sparse)"
basin_id = "03281500"
formula_path = "src/result/formulas/$(model_name)/$(basin_id)/qnn"
layer_eqs = extract_layer_eqs(formula_path)

@variables x1
# 0.81
q_func_121, params_121 = parse_to_func(layer_eqs["layer1_I2_A1"][6, :Equation], params_nm = :p121)
q_func_112, params_112 = parse_to_func(layer_eqs["layer1_I1_A2"][6, :Equation], params_nm = :p112)
q_func_122, params_122 = parse_to_func(layer_eqs["layer1_I2_A2"][8, :Equation], params_nm = :p122) # 8 for plot
q_func_132, params_132 = parse_to_func(layer_eqs["layer1_I3_A2"][6, :Equation], params_nm = :p132)
q_func_123, params_123 = parse_to_func(layer_eqs["layer1_I2_A3"][6, :Equation], params_nm = :p123)
# layer 2
q_func_211, params_211 = parse_to_func(layer_eqs["layer2_I1_A1"][9, :Equation], params_nm = :p211)
q_func_221, params_221 = parse_to_func(layer_eqs["layer2_I2_A1"][7, :Equation], params_nm = :p221)
q_func_231, params_231 = parse_to_func(layer_eqs["layer2_I3_A1"][8, :Equation], params_nm = :p231)

@info layer_eqs["layer1_I2_A3"][6, :Loss]
@info layer_eqs["layer1_I2_A3"][6, :Complexity]
q_params = ComponentArray(reduce(merge, (params_121, params_122, params_132, params_123, params_211, params_221, params_231)))

function symbolic_q_kan(x, p)
	acts1_output = q_func_121(x[2], p)
	acts2_output = q_func_122(x[2], p) + q_func_132(x[3], p)
	acts3_output = q_func_123(x[2], p)
	return q_func_211(acts1_output, p) + q_func_221(acts2_output, p) + q_func_231(acts3_output, p)
end

function load_splines(basin_id)
	q_nn_input = load_nn_data(basin_id, "exphydro(disc,withst)", "q")
	q_pas = load("src/result/models/$(model_name)/$(basin_id)/train_records.jld2")["opt_ps"].q
	q_nn = build_Q_NN(size(q_pas.layer_1.C)[1], 6)
	q_nn_states = LuxCore.initialstates(StableRNG(1234), q_nn)
	qnn_layer1_postacts = activation_getter(q_nn.layer_1, q_pas.layer_1, q_nn_states.layer_1, q_nn_input)
	qnn_layer1_postacts_arr = reduce((m1, m2) -> cat(m1, m2, dims = 3), qnn_layer1_postacts)
	qnn_layer1_output = q_nn.layer_1(q_nn_input, q_pas.layer_1, q_nn_states.layer_1)[1]
	qnn_layer2_postacts = activation_getter(q_nn.layer_2, q_pas.layer_2, q_nn_states.layer_2, qnn_layer1_output)
	qnn_layer2_postacts_arr = reduce((m1, m2) -> cat(m1, m2, dims = 3), qnn_layer2_postacts)
	return (q_nn_input, qnn_layer1_output), (qnn_layer1_postacts_arr, qnn_layer2_postacts_arr)
end

function plot_inner(input; color = 1)
	# Calculate predictions and R2
	min_val = minimum(input)
	max_val = maximum(input)
	mid_val = (min_val + max_val) / 2
	
	fig = plot(1:length(input), input,
		color = color,
		size = (300, 260), legend = false,
		grid = false, dpi=300,
		margin = 0Plots.mm, background_color = :transparent,
		yticks = ([min_val, mid_val, max_val], ["$(round(min_val,digits=2))", "$(round(mid_val,digits=2))", "$(round(max_val,digits=2))"]),
		xticks = nothing,
        tickfontsize = 18,
        fontfamily = "Times",
		framestyle = :box,
		linewidth = 3,
	)

	return fig
end

function plot_q_symbolize_fit(range = :)
	basin_id = "03281500"
	qnn_acts, qnn_postacts = load_splines(basin_id)
	q_input, qnn_layer1_output = qnn_acts

	fig_input1 = plot_inner(q_input[1, range], color = 1)
	fig_input2 = plot_inner(q_input[2, range], color = 2)
	fig_input3 = plot_inner(q_input[3, range], color = 3)

	# Layer 1 plots
	fig112 = plot_inner(q_func_112.(q_input[1, range], Ref(params_112)), color = 1)
	fig121 = plot_inner(q_func_121.(q_input[2, range], Ref(params_121)), color = 2)
	fig122 = plot_inner(q_func_122.(q_input[2, range], Ref(params_122)), color = 2)
	fig132 = plot_inner(q_func_132.(q_input[3, range], Ref(params_132)), color = 3)
	fig123 = plot_inner(q_func_123.(q_input[2, range], Ref(params_123)), color = 2)

    layer2_input1 = q_func_121.(q_input[2, range], Ref(params_121))
    layer2_input2 = q_func_112.(q_input[2, range], Ref(params_112)) .+ q_func_122.(q_input[2, range], Ref(params_122)) + q_func_132.(q_input[3, range], Ref(params_132))
    layer2_input3 = q_func_123.(q_input[2, range], Ref(params_123))

    layer2_input1_plot = plot_inner(layer2_input1, color = 5)
    layer2_input2_plot = plot_inner(layer2_input2, color = 5)
    layer2_input3_plot = plot_inner(layer2_input3, color = 5)

	# Layer 2 plots  
	fig211 = plot_inner(q_func_211.(qnn_layer1_output[1, range], Ref(params_211)), color = 4)
	fig221 = plot_inner(q_func_221.(qnn_layer1_output[2, range], Ref(params_221)), color = 4)
	fig231 = plot_inner(q_func_231.(qnn_layer1_output[3, range], Ref(params_231)), color = 4)

    output = q_func_211.(qnn_layer1_output[1, range], Ref(params_211)) + q_func_221.(qnn_layer1_output[2, range], Ref(params_221)) + q_func_231.(qnn_layer1_output[3, range], Ref(params_231))
    
    output_plot = plot_inner(output, color = :black)
	# Combine plots
	input_plots = [fig_input1, fig_input2, fig_input3]

    layer2_input_plots = [layer2_input1_plot, layer2_input2_plot, layer2_input3_plot]
	layer1_plots = [fig112, fig121, fig122, fig132, fig123]
	layer2_plots = [fig211, fig221, fig231]
	return input_plots, layer1_plots, layer2_input_plots, layer2_plots, output_plot


end

basin_id = "03281500"
qnn_acts, qnn_postacts = load_splines(basin_id)

input_plots, layer1_plots, layer2_input_plots, layer2_plots, output_plot = plot_q_symbolize_fit()
for (i, input_plot) in enumerate(input_plots)
	savefig(input_plot, "src/paper/figures/figure_11/input_plot_$(i).png")
end

for (i, layer1_plot) in enumerate(layer1_plots)
	savefig(layer1_plot, "src/paper/figures/figure_11/layer1_plot_$(i).png")
end

for (i, layer2_input_plot) in enumerate(layer2_input_plots)
	savefig(layer2_input_plot, "src/paper/figures/figure_11/layer2_input_plot_$(i).png")
end


for (i, layer2_plot) in enumerate(layer2_plots)
	savefig(layer2_plot, "src/paper/figures/figure_11/layer2_plot_$(i).png")
end

savefig(output_plot, "src/paper/figures/figure_11/output_plot.png")
