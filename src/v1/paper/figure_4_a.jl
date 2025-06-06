#* 图四:用于展示正则化模型NNq各节点的评分
using JLD2, DataFrames, CSV, Plots
include("../utils/train.jl")
include("../models/model_f.jl")
include("../utils/data_relate.jl")
include("../utils/kan_tools.jl")

basin_ids = ["02361000", "03281500", "06191500"]
model_name = "k50_base(reg)"

original_q_nn = build_Q_NN(6, 6)
model_layers = [original_q_nn.layer_1, original_q_nn.layer_2]

min_postact_scores = []
for basin_id in basin_ids
    kan_ckpt = load("src/result/models/$model_name/$basin_id/train_records.jld2")["opt_ps"]
    layer_params = kan_ckpt["q"]
    q_nn_input = load_nn_data(basin_id, "exphydro(disc,withst)", "q")
    splines_dict = obtain_nn_splines(model_layers=model_layers, layer_params=layer_params, input_data=q_nn_input)

    postacts1, postacts2 = splines_dict["postacts1"], splines_dict["postacts2"]
    max_postacts1 = vec(maximum(maximum(abs.(postacts1), dims = 1)[1, :, :], dims = 2))
    max_postacts2 = vec(maximum(abs.(postacts2), dims = 1)[1, :, :])
    min_postact_score = vec(minimum([max_postacts1 max_postacts2], dims = 2))
    push!(min_postact_scores, min_postact_score)
end

plots = []
palette1 = [:skyblue, :salmon, :mediumseagreen]
for (i, min_postact_score) in enumerate(min_postact_scores)
    colors = [palette1[i] for _ in 1:length(min_postact_score)]
    colors[min_postact_score .< 1.0] .= :gray
    basin_id = basin_ids[i]
    p = bar(min_postact_score,
        label="",
        color=colors,
        ylabel=i==1 ? "Maximum Activation Value" : "",
        xlabel="Node Index", 
        title="$(basin_id[1]=='0' ? basin_id[2:end] : basin_id)",
        dpi=300,
        fontfamily="Times",
        tickfontsize=12,
        guidefontsize=14,
        titlefontsize=14,
        margin=0.0Plots.mm)
        
    hline!([1.0],
        color=:black,
        linestyle=:dash,
        linewidth=2,
        label="")
        
    push!(plots, p)
end

plot(plots..., 
    layout=(1,3), 
    size=(1000,350),
    left_margin=10Plots.mm,
    bottom_margin=10Plots.mm,
    plot_title="",
    plot_titlefontsize=0)
savefig("src/paper/figures/figure_4_a.png")

