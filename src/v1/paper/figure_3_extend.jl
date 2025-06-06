# 绘制直方图,比较k50-f和m50-f的预测结果
using CSV, DataFrames, Plots, Statistics

r2_score = (x1, x2) -> 1 - sum((x1 .- x2).^2) / sum((x1 .- mean(x1)).^2)

model_name_list = [ "exphydro(disc,withst)","m50_base", "k50_base"] # "m50-p", "m50-f", "d-hbv", "k50-f", "k50-p", "hbv" "exphydro(cont2,withst)"
show_model_name = ["Exp-Hydro","M50", "K50"]

exphydro_disc_withst_predict_stats = CSV.read("src/result/stats/exphydro(disc,withst)-criteria.csv", DataFrame)
m50_base_predict_stats = CSV.read("src/result/stats/m50_base-criteria.csv", DataFrame)
k50_base_predict_stats = CSV.read("src/result/stats/k50_base-criteria.csv", DataFrame)

exphydro_disc_withst_predict_stats[!, Symbol("nse-test")][exphydro_disc_withst_predict_stats[!, Symbol("nse-test")] .< -1] .= -1
m50_base_predict_stats[!, Symbol("nse-test")][m50_base_predict_stats[!, Symbol("nse-test")] .< -1] .= -1
k50_base_predict_stats[!, Symbol("nse-test")][k50_base_predict_stats[!, Symbol("nse-test")] .< -1] .= -1

r2_score(exphydro_disc_withst_predict_stats[!, Symbol("nse-test")], m50_base_predict_stats[!, Symbol("nse-test")])
r2_score(exphydro_disc_withst_predict_stats[!, Symbol("nse-test")], k50_base_predict_stats[!, Symbol("nse-test")])

plot(exphydro_disc_withst_predict_stats[!, Symbol("nse-test")], m50_base_predict_stats[!, Symbol("nse-test")], seriestype=:scatter, label="M50", xlabel="Exphydro(disc,withst)", ylabel="M50", title="NSE")
plot(exphydro_disc_withst_predict_stats[!, Symbol("nse-test")], k50_base_predict_stats[!, Symbol("nse-test")], seriestype=:scatter, label="K50", xlabel="Exphydro(disc,withst)", ylabel="K50", title="NSE")
