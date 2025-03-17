# 绘制,符号化,原版以及ExpHydro模型的预测结果
using Plots
using DataFrames
using CSV
using Statistics
include("../utils/criteria.jl")

# 读取数据
function plot_figure_10(basin_id, show_ylabel=true)
    symbolize_result = CSV.read("src/result/models/k50_base(symbolize)/$(basin_id)/test_predicted_df.csv", DataFrame)
    exp_hydro_result = CSV.read("src/result/models/exphydro(disc,withst)/$(basin_id)/test_predicted_df.csv", DataFrame)
    k50_result = CSV.read("src/result/models/k50_base/$(basin_id)/test_predicted_df.csv", DataFrame)

    # 准备数据
    obs = symbolize_result.obs
    pred_symbolize = symbolize_result.pred 
    pred_exphydro = exp_hydro_result.pred
    pred_k50 = k50_result.pred

    nse_symbolize, mnse_symbolize, fhv_symbolize = nse(obs, pred_symbolize), mnse(obs, pred_symbolize), fhv(obs, pred_symbolize)
    nse_exphydro, mnse_exphydro, fhv_exphydro = nse(obs, pred_exphydro), mnse(obs, pred_exphydro), fhv(obs, pred_exphydro)
    nse_k50, mnse_k50, fhv_k50 = nse(obs, pred_k50), mnse(obs, pred_k50), fhv(obs, pred_k50)
    basin_id = basin_id[1]=='0' ? basin_id[2:end] : basin_id
    # 创建子图
    p1 = plot(1:length(obs), [obs pred_symbolize], 
        label=["Observed($basin_id)" "K50(Symbolize)"],
        color=[:black 1],
        linestyle=[:solid :solid],
        linewidth=[1 2],
        alpha=[1.0 0.8],
        fontfamily="Times",
        xlabel="Time Index (day)",
        ylabel=show_ylabel ? "Runoff (mm/d)" : "",
        tickfont=font(12, "Times"),
        guidefont=font(14, "Times"),
        titlefont=font(16, "Times"),
        legendfont=font(12, "Times"),
        size=(500, 300),
        dpi=300)
    plot!(p1, legend=:topright)

    annote_pos1 = (0.3*length(obs), 0.6*maximum(obs))
    annotate!(p1, [(annote_pos1..., Plots.text("NSE = $(round(nse_symbolize, digits=2))\nmNSE = $(round(mnse_symbolize, digits=2))\nFHV = $(round(fhv_symbolize, digits=2))", 14, "Times", :right, :bottom, color=:black))])

    p2 = plot(1:length(obs), [obs pred_exphydro],
        label=["Observed($basin_id)" "Exp-Hydro"],
        color=[:black 2],
        linestyle=[:solid :solid], 
        linewidth=[1 2],
        alpha=[1.0 0.8],
        fontfamily="Times",
        ylabel=show_ylabel ? "Runoff (mm/d)" : "",
        tickfont=font(12, "Times"),
        guidefont=font(14, "Times"),
        titlefont=font(16, "Times"),
        legendfont=font(12, "Times"),
        size=(500, 300),
        dpi=300)
    plot!(p2, legend=:topright)

    annote_pos2 = (0.3*length(obs), 0.6*maximum(obs))
    annotate!(p2, [(annote_pos2..., Plots.text("NSE = $(round(nse_exphydro, digits=2))\nmNSE = $(round(mnse_exphydro, digits=2))\nFHV = $(round(fhv_exphydro, digits=2))", 14, "Times", :right, :bottom, color=:black))])

    p3 = plot(1:length(obs), [obs pred_k50],
        label=["Observed($basin_id)" "K50"],
        color=[:black 3],
        linestyle=[:solid :solid],
        linewidth=[1 2], 
        alpha=[1.0 0.8],
        fontfamily="Times",
        ylabel=show_ylabel ? "Streamflow (mm/d)" : "",
        tickfont=font(12, "Times"),
        guidefont=font(14, "Times"),
        titlefont=font(16, "Times"),
        legendfont=font(12, "Times"),
        size=(500, 300),
        dpi=300)
        
    plot!(p3, legend=:topright)

    annote_pos3 = (0.3*length(obs), 0.6*maximum(obs))
    annotate!(p3, [(annote_pos3..., Plots.text("NSE = $(round(nse_k50, digits=2))\nmNSE = $(round(mnse_k50, digits=2))\nFHV = $(round(fhv_k50, digits=2))", 14, "Times", :right, :bottom, color=:black))])
        

    # 组合子图
    combined_plot = plot(p2, p3,p1, 
        layout=(3,1),
        size=(500, 900),
        left_margin=6Plots.mm,
        bottom_margin=6Plots.mm,
        margin=1Plots.mm)
    combined_plot
end

combine_1 = plot_figure_10("02361000",true)
combine_2 = plot_figure_10("03281500",false)
combine_3 = plot_figure_10("06191500",false)

total_plot = plot(combine_1, combine_2, combine_3, layout=(1,3), size=(1500, 900))
savefig(total_plot, "src/paper/figures/figure_10.png")
