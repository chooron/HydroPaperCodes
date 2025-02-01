using Lux
using JLD2
using SymbolicRegression
using Symbolics
using SymbolicUtils

include("../utils/kan_tools.jl")
include("../models/model_f.jl")
include("../utils/symbolic_relate.jl")
include("../utils/data_relate.jl")

refit = true

basin_id = "03281500" # call by other file
model_name = "k50(sparse)"
model_dir = "src/result/$(model_name)/$(basin_id)"
kan_ckpt = load("$(model_dir)/pruned_train_records.jld2")["opt_ps"]
et_pas, q_pas = kan_ckpt["et"], kan_ckpt["q"]

et_nn_dims = size(et_pas.layer_1.C)[1]
@info "pruned et_nn_dims: $et_nn_dims"
et_nn = build_ET_NN(et_nn_dims, 6)
et_input = load_nn_data(basin_id, "exphydro(disc,withst)", "et")

et_nn_states = LuxCore.initialstates(StableRNG(1234), et_nn)
etnn_layer1_postacts = activation_getter(et_nn.layer_1, et_pas.layer_1, et_nn_states.layer_1, et_input)
etnn_layer1_postacts_arr = reduce((m1, m2) -> cat(m1, m2, dims = 3), etnn_layer1_postacts)

etnn_layer1_output = et_nn.layer_1(et_input, et_pas.layer_1, et_nn_states.layer_1)[1]
etnn_layer2_postacts = activation_getter(et_nn.layer_2, et_pas.layer_2, et_nn_states.layer_2, etnn_layer1_output)
etnn_layer2_postacts_arr = reduce((m1, m2) -> cat(m1, m2, dims = 3), etnn_layer2_postacts)

p1 = plot_activation_functions(et_input, etnn_layer1_postacts_arr)
p2 = plot_activation_functions(etnn_layer1_output, etnn_layer2_postacts_arr)
mkpath("src/plots/$(model_name)/$(basin_id)")
savefig(p1, "src/plots/$(model_name)/$(basin_id)/etnn_layer1_acts.png")
savefig(p2, "src/plots/$(model_name)/$(basin_id)/etnn_layer2_acts.png")

if refit == true
	fit_kan_activations(et_nn, et_input, et_pas, "etnn")
end

submodel_name = "etnn"
formula_path = "src/result/formulas/$(basin_id)/$(submodel_name)"
et_layer_eqs = extract_layer_eqs(formula_path)
@variables x1
#* 将字符串的公式转换为function得到对应的参数值
# layer 1
et_func_122, et_params_122 = parse_to_func(et_layer_eqs["layer1_I2_A2"][end, :Equation], params_nm=:p122)
et_func_132, et_params_132 = parse_to_func(et_layer_eqs["layer1_I3_A2"][end, :Equation], params_nm=:p132)
et_func_141, et_params_141 = parse_to_func(et_layer_eqs["layer1_I4_A1"][end, :Equation], params_nm=:p141)
et_func_142, et_params_142 = parse_to_func(et_layer_eqs["layer1_I4_A2"][end, :Equation], params_nm=:p142)
# layer 2
et_func_211, et_params_211 = parse_to_func(et_layer_eqs["layer2_I1_A1"][end, :Equation], params_nm=:p211)
et_func_221, et_params_221 = parse_to_func(et_layer_eqs["layer2_I2_A1"][end, :Equation], params_nm=:p221)

et_nn_states = LuxCore.initialstates(StableRNG(1234), et_nn)
etnn_layer1_postacts = activation_getter(et_nn.layer_1, et_pas.layer_1, et_nn_states.layer_1, et_input)
etnn_layer1_postacts_arr = reduce((m1, m2) -> cat(m1, m2, dims = 3), etnn_layer1_postacts)

etnn_layer1_output = et_nn.layer_1(et_input, et_pas.layer_1, et_nn_states.layer_1)[1]
etnn_layer2_postacts = activation_getter(et_nn.layer_2, et_pas.layer_2, et_nn_states.layer_2, etnn_layer1_output)
etnn_layer2_postacts_arr = reduce((m1, m2) -> cat(m1, m2, dims = 3), etnn_layer2_postacts)

function plot_etnn_symbolize_fit()
	fig111 = check_fit_plot(et_func_111, et_params_111, et_input[1,:], etnn_layer1_postacts_arr[:,1,1])
	fig121 = check_fit_plot(et_func_121, et_params_121, et_input[2,:], etnn_layer1_postacts_arr[:,1,2])
	fig131 = check_fit_plot(et_func_131, et_params_131, et_input[3,:], etnn_layer1_postacts_arr[:,1,3])
	fig141 = check_fit_plot(et_func_141, et_params_141, et_input[4,:], etnn_layer1_postacts_arr[:,1,4])
	fig112 = check_fit_plot(et_func_112, et_params_112, et_input[1,:], etnn_layer1_postacts_arr[:,2,1])
	fig122 = check_fit_plot(et_func_122, et_params_122, et_input[2,:], etnn_layer1_postacts_arr[:,2,2])
	fig132 = check_fit_plot(et_func_132, et_params_132, et_input[3,:], etnn_layer1_postacts_arr[:,2,3])
	fig142 = check_fit_plot(et_func_142, et_params_142, et_input[4,:], etnn_layer1_postacts_arr[:,2,4])

	fig211 = check_fit_plot(et_func_211, et_params_211, etnn_layer1_output[1,:], etnn_layer2_postacts_arr[:,1,1])
	fig221 = check_fit_plot(et_func_221, et_params_221, etnn_layer1_output[2,:], etnn_layer2_postacts_arr[:,1,2])
	return (fig111, fig112, fig121, fig122, fig131, fig132, fig141, fig142, fig211, fig221)
end












