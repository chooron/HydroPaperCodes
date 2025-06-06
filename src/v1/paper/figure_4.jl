# 绘制qnn训练前后变化的预测性能变化
using Plots
using Statistics
using DataFrames, CSV, DelimitedFiles

m50_posttrain_result = CSV.read("src/result/stats/m50_base-criteria.csv", DataFrame)
k50_posttrain_result = CSV.read("src/result/stats/k50_base-criteria.csv", DataFrame)

function plot_cummulate_fig(posttrain_result, criteria, show_label = true, label_name = "")
	m50_posttrain_result, k50_posttrain_result = posttrain_result


	m50_posttrain_train_nse = m50_posttrain_result[!, "$criteria-train"]
	k50_posttrain_train_nse = k50_posttrain_result[!, "$criteria-train"]

	m50_posttrain_test_nse = m50_posttrain_result[!, "$criteria-test"]
	k50_posttrain_test_nse = k50_posttrain_result[!, "$criteria-test"]

	# Calculate median values
	m50_train_median = median(m50_posttrain_train_nse)
	k50_train_median = median(k50_posttrain_train_nse)
	m50_test_median = median(m50_posttrain_test_nse)
	k50_test_median = median(k50_posttrain_test_nse)

	m50_posttrain_train_nse[m50_posttrain_train_nse.<-1] .= -1
	k50_posttrain_train_nse[k50_posttrain_train_nse.<-1] .= -1

	m50_posttrain_test_nse[m50_posttrain_test_nse.<-1] .= -1
	k50_posttrain_test_nse[k50_posttrain_test_nse.<-1] .= -1

	sorted_m50_posttrain_train = sort(m50_posttrain_train_nse)
	sorted_k50_posttrain_train = sort(k50_posttrain_train_nse)

	sorted_m50_posttrain_test = sort(m50_posttrain_test_nse)
	sorted_k50_posttrain_test = sort(k50_posttrain_test_nse)

	# Calculate cumulative probabilities
	n = length(m50_posttrain_train_nse)
	cum_prob_train = collect(1:n) ./ n
	cum_prob_test = collect(1:n) ./ n


	# Create cumulative distribution plot
	p = plot(sorted_m50_posttrain_train, cum_prob_train, label = "M50-train",
		linewidth = 2, dpi = 300, size = (300, 300), fontfamily="Times", 
	)

	plot!(
		sorted_k50_posttrain_train, cum_prob_train, label = "K50-train", 
		linewidth = 2, fontfamily="Times"
	)

	plot!(
		sorted_m50_posttrain_test, cum_prob_test, label = "M50-test",
		linewidth = 2
	)

	plot!(
		sorted_k50_posttrain_test, cum_prob_test, label = "K50-test",
		linewidth = 2
	)

	# Add median value lines with matching colors and annotations
	vline!([m50_train_median], label="M50-train median $(round(m50_train_median, digits=3))", linestyle=:dot, color=1, linewidth=2, fontfamily="Times")

	vline!([k50_train_median], label="K50-train median $(round(k50_train_median, digits=3))", linestyle=:dot, color=2, linewidth=2, fontfamily="Times") 

	vline!([m50_test_median], label="M50-test median $(round(m50_test_median, digits=3))", linestyle=:dot, color=3, linewidth=2, fontfamily="Times")

	vline!([k50_test_median], label="K50-test median $(round(k50_test_median, digits=3))", linestyle=:dot, color=4, linewidth=2, fontfamily="Times")

	xlabel!("$label_name", fontsize=12, fontfamily="Times")
	if show_label
		ylabel!("Cumulative Probability", fontsize=12, fontfamily="Times")
	end
	plot!(framestyle = :box, grid = false)
	plot!(legend=:topleft, fontsize=12, fontfamily="Times")
	plot!(xticks=:auto, yticks=:auto, xtickfontsize=10, ytickfontsize=10)
	p
end

p1 = plot_cummulate_fig([m50_posttrain_result, k50_posttrain_result], "nse", true, "NSE")
p2 = plot_cummulate_fig([m50_posttrain_result, k50_posttrain_result], "mnse", false, "mNSE")

p = plot(p1, p2, layout=(1,2), size=(700,300), dpi=300, left_margin=3Plots.mm, bottom_margin=3Plots.mm)
savefig(p, "src/paper/figures/figure_4.png")

