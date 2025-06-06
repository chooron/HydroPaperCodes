using CSV, DataFrames, Dates
using Plots, Statistics, JLD2, Shapefile, Measures

shp_path = "data/gis/us-main.shp"
station_position = CSV.read("data/gis/gauge_info.csv", DataFrame)

for method in ["ig", "sg"]
    for model in ["kan2"]
        for (var, colorbar, color) in zip(["S0", "S1", "I"], [:Blues, :Reds, :Greens], [:skyblue, :salmon, :mediumseagreen])
            xai_result = CSV.read("result/xai/camels_$(method)_values_$(model).csv", DataFrame)
            # 筛选出只在station_xai_result出现的station_position
            select_position = station_position[in.(station_position[!, :GAGE_ID], Ref(xai_result[!, :basin_id])), :]
            station_lat = select_position[!, :LAT]
            station_lon = select_position[!, :LON]
            xai_total = sum(xai_result[!, [:S0, :S1, :I]] |> Array, dims=2)
            xai_result = xai_result[!, var] ./ xai_total

            # 计算min和max以缩短colorbar
            min_val = 0.0
            max_val = 1.0
            # 创建简洁的刻度标签
            tick_labels = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]

            # 加载 SHP 文件
            table = Shapefile.Table(shp_path)
            shapes = Shapefile.shapes(table)

            # 创建绘图
            p1 = plot(xlabel="longitude", ylabel="latitude", aspect_ratio=:equal, legend=false,
                 xticks=nothing, yticks=nothing, grid=false, framestyle=:none, 
                 dpi=900,
                 fontfamily="Times")

            # 绘制 SHP 文件中的地理边界
            for shape in shapes
                plot!(p1, shape, fillalpha=0.5, linecolor=:darkgrey, fillcolor=:lightgrey, linewidth=0.5)
            end

            # 绘制站点位置
            scatter!(p1, station_lon, station_lat,
                marker_z=xai_result,
                color=colorbar,
                markersize=3,
                markerstrokewidth=0.1,
                markerstrokecolor=:gray,
                colorbar=true,
                # colorbar_ticks=([0.0, 1.0], tick_labels),
                colorbar_tickfontsize=14,
                clims=(0.0, 1.0))
            # 保存两个图
            savefig(p1, "result/xai/plots/camels_$(method)_$(model)_$(var)_map.png")
        end
    end
end

for method in ["ig", "sg"]
    for (var, color) in zip(["S0", "S1", "I"], [:skyblue, :salmon, :mediumseagreen])
        mlp_xai_result = CSV.read("result/xai/camels_$(method)_values_mlp.csv", DataFrame)
        # 筛选出只在station_xai_result出现的station_position
        mlp_xai_total = sum(mlp_xai_result[!, [:S0, :S1, :I]] |> Array, dims=2)
        mlp_xai_result = mlp_xai_result[!, var] ./ mlp_xai_total

        kan_xai_result = CSV.read("result/xai/camels_$(method)_values_kan.csv", DataFrame)
        kan_xai_total = sum(kan_xai_result[!, [:S0, :S1, :I]] |> Array, dims=2)
        kan_xai_result = kan_xai_result[!, var] ./ kan_xai_total

        # 创建直方图
        bins = range(0, 1, length=21)  # 创建20个等距的bin，从0到1
        p2 = histogram(mlp_xai_result, 
            bins=bins,
            dpi=300,
            size=(700, 500),
            color=:darkgrey,
            alpha=0.8,
            label="MLP",
            legend=:topright,
            fontsize=14,
            fontfamily="Times",
            xlabel="Importance",
            ylabel="Count",
            bottom_margin=10mm,
            xlabelfontsize=18,
            ylabelfontsize=18,
            tickfontsize=14)

        histogram!(p2, kan_xai_result,
            bins=bins,
            color=color,
            alpha=0.8,
            label="KAN",
            fontsize=14,
            legendfontsize=14,
            fontfamily="Times")

        savefig(p2, "result/xai/plots/camels_$(method)_$(var)_hist.png")
    end
end
