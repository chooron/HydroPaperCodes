using Symbolics
using ComponentArrays
using Plots
using Statistics
using Lux
using StableRNGs

include("../../../utils/symbolic_relate.jl")
include("../../../utils/kan_tools.jl")
include("../../../utils/data_relate.jl")
include("../../../models/model_f.jl")

model_name = "k50_base(sparse)"
basin_id = "06191500"
formula_path = "src/result/formulas/$(model_name)/$(basin_id)/qnn"
layer_eqs = extract_layer_eqs(formula_path)

q_func_121, params_121 = parse_to_func(layer_eqs["layer1_I2_A1"][8, :Equation], params_nm = :p121)
q_func_122, params_122 = parse_to_func(layer_eqs["layer1_I2_A2"][7, :Equation], params_nm = :p122)
# layer 2
q_func_211, params_211 = parse_to_func(layer_eqs["layer2_I1_A1"][8, :Equation], params_nm = :p211)
q_func_221, params_221 = parse_to_func(layer_eqs["layer2_I2_A1"][10, :Equation], params_nm = :p221)

q_params = ComponentArray(reduce(merge, (params_121, params_122, params_211, params_221)))

function symbolic_q_kan(x, p)
	acts1_output = q_func_121(x[2], p) 
	acts2_output = q_func_122(x[2], p)
	return q_func_211(acts1_output, p) + q_func_221(acts2_output, p)
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
function check_fit_plot(tmp_func, params, input, target; title = "", figsize = (400, 400))
	# Sort input and get sorting indices
	sorted_indices = sortperm(input)
	sorted_target = target[sorted_indices]
	sorted_input = input[sorted_indices]

	# Calculate predictions and R2
	predictions = tmp_func.(sorted_input, Ref(params))
	ss_res = sum((sorted_target .- predictions) .^ 2)
	ss_tot = sum((sorted_target .- mean(sorted_target)) .^ 2)
	r2 = 1 - ss_res / ss_tot

	fig = plot(size = figsize)
	# Plot predicted line
	plot!(sorted_input, predictions, fontfamily="Times", fontsize=14, guidefontsize=16, tickfont=font(14, "Times"),dpi=300,
		linewidth = 4, color = :blue, label = "Predicted", legendfontsize=14, legendfontfamily="Times", bottom_margin=5Plots.mm)
	# Plot target points  
	scatter!(sorted_input, sorted_target, dpi=300,
		color = :black, markersize = 3, label = "Target", alpha = 0.4)
	# Add title with R2
	title!("$(title)", fontsize=16, fontfamily="Times")
	# Add R² annotation in bottom right corner
	bound_v = max(maximum(predictions), maximum(sorted_target)) - min(minimum(predictions), minimum(sorted_target))
	annote_pos = (maximum(sorted_input), max(maximum(predictions), maximum(sorted_target)) - 0.9*bound_v)
	annotate!([(annote_pos..., Plots.text("R² = $(round(r2, digits=2))", 16, "Times", :right, :bottom))])

	return fig
end

function plot_q_symbolize_fit()
	basin_id = "06191500"
	qnn_acts, qnn_postacts = load_splines(basin_id)
	q_input, qnn_layer1_output = qnn_acts
	qnn_layer1_postacts_arr, qnn_layer2_postacts_arr = qnn_postacts

	# Layer 1 plots
	fig121 = check_fit_plot(q_func_121, params_121, q_input[2, :], qnn_layer1_postacts_arr[:, 1, 2], title = "𝛷(1,2,1)", figsize = (300, 300))
	fig122 = check_fit_plot(q_func_122, params_122, q_input[2, :], qnn_layer1_postacts_arr[:, 2, 2], title = "𝛷(1,2,2)", figsize = (300, 300))

	# Layer 2 plots  
	fig211 = check_fit_plot(q_func_211, params_211, qnn_layer1_output[1, :], qnn_layer2_postacts_arr[:, 1, 1], title = "𝛷(2,1,1)", figsize = (300, 300))
	fig221 = check_fit_plot(q_func_221, params_221, qnn_layer1_output[2, :], qnn_layer2_postacts_arr[:, 1, 2], title = "𝛷(2,2,1)", figsize = (300, 300))

	# Combine plots
	layer1_plot = plot(fig121, fig122, layout = (1, 2), size = (600, 300))
	layer2_plot = plot(fig211, fig221, layout = (1, 2), size = (600, 300))
	return layer1_plot, layer2_plot
end

# p1, p2 = plot_q_symbolize_fit()
# mkpath("src/paper/figures/figure_9/$(basin_id)")
# savefig(p1, "src/paper/figures/figure_9/$(basin_id)/qnn_layer1_acts.png")
# savefig(p2, "src/paper/figures/figure_9/$(basin_id)/qnn_layer2_acts.png")