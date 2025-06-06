using CSV, DataFrames, Plots
include("figure_5.b.jl")

basin_ids = ["02361000", "03281500", "06191500"]
base_dir = "src/result/models/"

k50_loss_df_list = DataFrame[]
m50_loss_df_list = DataFrame[]
for basin_id in basin_ids
	m50_basin_dir = joinpath(base_dir, "m50_base", basin_id)
	k50_basin_dir = joinpath(base_dir, "k50_base", basin_id)
	loss_df_k50 = CSV.read(joinpath(k50_basin_dir, "loss_df.csv"), DataFrame)
	loss_df_m50 = CSV.read(joinpath(m50_basin_dir, "loss_df.csv"), DataFrame)
	push!(k50_loss_df_list, loss_df_k50)
	push!(m50_loss_df_list, loss_df_m50)
end

# Create a list to store individual plots
plots_list = []

# Create individual plots
for (i, (loss_df_k50, loss_df_m50)) in enumerate(zip(k50_loss_df_list, m50_loss_df_list))
    local p = plot(loss_df_k50.train_loss,
            label="",
            color=1, 
            linewidth=2,
            size=(500,300),)
    plot!(p, loss_df_k50.val_loss,
            label="",
            color=2, 
            linewidth=2)
    plot!(p, loss_df_m50.train_loss,
            label="",
            color=3, 
            linewidth=2)
    plot!(p, loss_df_m50.val_loss,
            label="",
            color=4, 
            linewidth=2)
    
    # 增加字体大小
    xlabel!(p, "Iteration", guidefontsize=18, fontfamily="Times")
    ylabel!(p, "Loss", guidefontsize=18, fontfamily="Times")
    
    # 增加刻度字体大小
    plot!(p, xtickfont=font(16, "Times"),
          ytickfont=font(16, "Times"))
          
    push!(plots_list, p)
end

# Combine plots into one figure
p1 = plot(plots_list..., 
          layout=(3,1), 
          size=(500,900),    # 增加图形大小
          dpi=300,
          left_margin=10Plots.mm,
)

# Combine p1 and p2 side by side
final_plot = plot(p1, p2, 
                 layout=(1,2),
                 size=(1000,900), # Double the width to accommodate two plots
                 dpi=300,
                 left_margin=10Plots.mm)
savefig(final_plot, "src/paper/figures/figure_5/final_plot.png")
