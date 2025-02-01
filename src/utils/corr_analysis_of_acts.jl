# 分析激活曲线与原本模型计算公式的相关性
using Lux
using JLD2
using SymbolicRegression
using Symbolics
using SymbolicUtils

include("../utils/kan_tools.jl")
include("../models/model_f.jl")
include("../utils/symbolic_relate.jl")
include("../utils/data_relate.jl")

refit = false

basin_id = "03281500"  # 02361000, 03281500, 06191500  # call by other file
model_name = "k50(sparse)"
basemodel_name = "exphydro(disc,withst)"
model_dir = "src/result/$(model_name)/$(basin_id)"
kan_ckpt = load("$(model_dir)/pruned_train_records.jld2")["opt_ps"]
q_pas, q_pas = kan_ckpt["et"], kan_ckpt["q"]

q_nn_dims = size(q_pas.layer_1.C)[1]
@info "pruned q_nn_dims: $q_nn_dims"
q_nn = build_Q_NN(q_nn_dims, 6)
#* load data

q_input = load_nn_data(basin_id, basemodel_name, "q")

q_nn_states = LuxCore.initialstates(StableRNG(1234), q_nn)
qnn_layer1_postacts = activation_getter(q_nn.layer_1, q_pas.layer_1, q_nn_states.layer_1, q_input)
qnn_layer1_postacts_arr = reduce((m1, m2) -> cat(m1, m2, dims = 3), qnn_layer1_postacts)
qnn_layer1_output = q_nn.layer_1(q_input, q_pas.layer_1, q_nn_states.layer_1)[1]

exphydro_df = CSV.read("src/result/models/$basemodel_name/$(basin_id)/model_outputs.csv", DataFrame)
snowpack_vec, soilwater_vec = exphydro_df[!, "snowpack"], exphydro_df[!, "soilwater"]
soilwater_vec, qsim_vec = exphydro_df[!, "soilwater"], exphydro_df[!, "qsim"]
# Sort soilwater and qsim vectors
norm_soilwater_vec = (soilwater_vec .- minimum(soilwater_vec)) ./ (maximum(soilwater_vec) - minimum(soilwater_vec))
norm_qsim_vec = (qsim_vec .- minimum(qsim_vec)) ./ (maximum(qsim_vec) - minimum(qsim_vec))
sorted_soilwater = norm_soilwater_vec[sortperm(norm_soilwater_vec)]
sorted_qsim = norm_qsim_vec[sortperm(norm_soilwater_vec)]

# Create scatter plot
p = scatter(
    sorted_soilwater[1:10:end], sorted_qsim[1:10:end], # Only plot every 10th point
    xlabel="Soil Water Content", 
    ylabel="Streamflow", 
    label=false,
    alpha=0.3,
    markersize=2,
    fontfamily="Times",
    markerstrokecolor=:lightgray # Set marker border color to light gray
)

for i in axes(qnn_layer1_postacts_arr, 2)
    nn_soilwater_input = q_input[2, :]
    sorted_nn_soilwater = nn_soilwater_input[sortperm(nn_soilwater_input)]
    sorted_nn_qsim = qnn_layer1_postacts_arr[:, i, 2][sortperm(nn_soilwater_input)]
    norm_sorted_nn_soilwater = (sorted_nn_soilwater .- minimum(sorted_nn_soilwater)) ./ (maximum(sorted_nn_soilwater) - minimum(sorted_nn_soilwater))
    norm_sorted_nn_qsim = (sorted_nn_qsim .- minimum(sorted_nn_qsim)) ./ (maximum(sorted_nn_qsim) - minimum(sorted_nn_qsim))

    plot!(
        norm_sorted_nn_soilwater, norm_sorted_nn_qsim,
        xlabel="Soil Water Content", 
        ylabel="Streamflow",
        label=false,
        fontfamily="Times",
        alpha=0.5,
        linewidth=2
    )
end
p