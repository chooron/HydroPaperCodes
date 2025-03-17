# 绘制直方图,比较k50-f和m50-f的预测结果
using CSV, DataFrames, Plots, Statistics


function plot_model_stats(stats_df, model_name)
    # Filter out rows where nse-test < -1
    # stats_df = stats_df[stats_df[!, Symbol("nse-test")] .>= -1, :]

    # Extract metrics
    nse_test = stats_df[!, Symbol("nse-test")]
    mnse_test = stats_df[!, Symbol("mnse-test")] 
    fhv_test = stats_df[!, Symbol("fhv-test")]

    # Calculate statistics
    nse_mean = mean(nse_test)
    nse_median = median(nse_test)
    mnse_mean = mean(mnse_test)
    mnse_median = median(mnse_test)
    fhv_mean = mean(fhv_test)
    fhv_median = median(fhv_test)

    # Constrain NSE and KGE between -1 and 1
    nse_test[nse_test .< 0] .= 0
    mnse_test[mnse_test .< 0] .= 0
    fhv_test[fhv_test .> 75] .= 75
    
    # Create bins
    bins = 0:0.02:1
    
    # Create subplot layout with increased top margin and reduced spacing
    p = plot(layout=(1,3), size=(1000,200), top_margin=3Plots.mm, left_margin=5Plots.mm, dpi=300, margin=1*Plots.mm, bottom_margin=2Plots.mm)

    # Define 3 color palettes
    palette1 = (:skyblue, :salmon, :mediumseagreen)  # Soft and professional

    # Plot NSE histogram
    histogram!(p[1], nse_test, bins=bins, label="NSE", color=palette1[1], tickfontsize=12, legendfontsize=12, fontfamily="Times", legend=:topleft)
    vline!(p[1], [nse_mean], label="mean", color=:black, linewidth=2, fontfamily="Times")
    vline!(p[1], [nse_median], label="median", color=:grey, linestyle=:dash, linewidth=2, fontfamily="Times")
    annotate!(p[1], mean(xlims(p[1])), maximum(ylims(p[1]))*1.05, "mean=$(round(nse_mean,digits=3)), median=$(round(nse_median,digits=3))", fontsize=12, fontfamily="Times")
    
    # Plot KGE histogram  
    histogram!(p[2], mnse_test, bins=bins, label="mNSE", color=palette1[2], tickfontsize=12, legendfontsize=12, fontfamily="Times")
    vline!(p[2], [mnse_mean], label="mean", color=:black, linewidth=2, fontfamily="Times")
    vline!(p[2], [mnse_median], label="median", color=:grey, linestyle=:dash, linewidth=2, fontfamily="Times")
    annotate!(p[2], mean(xlims(p[2])), maximum(ylims(p[2]))*1.05, "mean=$(round(mnse_mean,digits=3)), median=$(round(mnse_median,digits=3))", fontsize=12, fontfamily="Times")
    
    # Plot FHV histogram
    histogram!(p[3], fhv_test, bins=0:2:75, label="FHV", color=palette1[3], tickfontsize=12, legendfontsize=12, fontfamily="Times")
    vline!(p[3], [fhv_mean], label="mean", color=:black, linewidth=2, fontfamily="Times")
    vline!(p[3], [fhv_median], label="median", color=:grey, linestyle=:dash, linewidth=2, fontfamily="Times")
    annotate!(p[3], mean(xlims(p[3])), maximum(ylims(p[3]))*1.05, "mean=$(round(fhv_mean,digits=3)), median=$(round(fhv_median,digits=3))", fontsize=12, fontfamily="Times")
    
    # Set y-axis label for leftmost plot with adjusted position
    ylabel!(p[1], model_name, labelfontsize=14, right_margin=5Plots.mm)
    return p
end

model_name_list = [ "exphydro(disc,withst)","m50_base", "k50_base"] # "m50-p", "m50-f", "d-hbv", "k50-f", "k50-p", "hbv" "exphydro(cont2,withst)"
show_model_name = ["Exp-Hydro","M50", "K50"]
plot_list = []
for (model_name, show_name) in zip(model_name_list, show_model_name)
    m50_predict_stats = CSV.read("src/result/stats/$model_name-criteria.csv", DataFrame)
    tmp_plot = plot_model_stats(m50_predict_stats, show_name)
    push!(plot_list, tmp_plot)
    savefig(tmp_plot, "src/paper/figures/figure_3/$(model_name)-hist.png")
end

plot(plot_list..., layout=grid(length(model_name_list), 1), size=(1000, 200*length(model_name_list)), dpi=300)
savefig("src/paper/figures/figure_3/combined_hist.png")