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
    et_pas, q_pas = kan_ckpt["et"], kan_ckpt["q"]
    
    et_nn_dims = size(et_pas.layer_1.C)[1]
    @info "pruned et_nn_dims: $et_nn_dims"
    et_nn = build_Q_NN(et_nn_dims, 6)
    #* load data
    
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
    
    fit_kan_activations(et_nn, et_input, et_pas, "src/result/formulas/$(model_name)/$(basin_id)/etnn")
end
