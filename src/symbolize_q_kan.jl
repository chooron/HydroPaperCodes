# 拟合KAN的激活曲线然后构建符号公式


using Lux
using JLD2
using SymbolicRegression
using Symbolics
using SymbolicUtils

include("utils/kan_tools.jl")
include("models/model_f.jl")
include("utils/symbolic_relate.jl")
include("utils/data_relate.jl")

# basin_id = "03281500"  # 02361000, 03281500, 06191500  # call by other file
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
    
    p1 = plot_activation_functions(q_input, qnn_layer1_postacts_arr)
    p2 = plot_activation_functions(qnn_layer1_output, qnn_layer2_postacts_arr)
    mkpath("src/plots/$(model_name)/$(basin_id)")
    savefig(p1, "src/plots/$(model_name)/$(basin_id)/qnn_layer1_acts.png")
    savefig(p2, "src/plots/$(model_name)/$(basin_id)/qnn_layer2_acts.png")
    
    fit_kan_activations(q_nn, q_input, q_pas, "src/result/formulas/$(model_name)/$(basin_id)/qnn")
end
