using JLD2
using DataFrames
using Plots
using DelimitedFiles
using Statistics

basin_file = readdlm(joinpath("data/basin_ids/basins_all.txt"), ',')
basins_available = lpad.(string.(Int.(basin_file[:, 1])), 8, "0")
reg_gamma = "5e-3"
gamma_dict = Dict("1e-2" => 1e-2, "1e-3" => 1e-3, "5e-3" => 5e-3)
data_list = []
for basin_id in basins_available
    result_path = "result/v2/k50_reg($reg_gamma)/$basin_id/loss_df.jld2"
    result = load(result_path)

    adam_loss_sorted = sort(result["adam_loss_df"], :train_loss)
    adam_best_train_loss = adam_loss_sorted[1, :train_loss]
    adam_best_val_loss = adam_loss_sorted[1, :val_loss]
    lbfgs_loss = result["lbfgs_loss_df"]
    init_reg_loss = lbfgs_loss[1, :reg_loss] ./ gamma_dict[reg_gamma]
    lbfgs_loss_sorted = sort(result["lbfgs_loss_df"], :val_loss)
    lbfgs_best_train_loss = lbfgs_loss_sorted[1, :train_loss]
    lbfgs_best_val_loss = lbfgs_loss_sorted[1, :val_loss]
    lbfgs_best_reg_loss = lbfgs_loss_sorted[1, :reg_loss] ./ gamma_dict[reg_gamma]


    tmp_loss_list = [basin_id, adam_best_train_loss, adam_best_val_loss, init_reg_loss, lbfgs_best_train_loss, lbfgs_best_val_loss, lbfgs_best_reg_loss]
    push!(data_list, NamedTuple{(:basin_id, :adam_train_loss, :adam_val_loss, :init_reg_loss, :lbfgs_train_loss, :lbfgs_val_loss, :lbfgs_reg_loss)}(tmp_loss_list))
end

data_df = DataFrame(data_list, [:basin_id, :adam_train_loss, :adam_val_loss, :init_reg_loss, :lbfgs_train_loss, :lbfgs_val_loss, :lbfgs_reg_loss])
CSV.write("src/v2/stats/k50_reg($reg_gamma)_loss_df.csv", data_df)
@info "lbfgs_train_loss: $(median(data_df.lbfgs_train_loss)), lbfgs_val_loss: $(median(data_df.lbfgs_val_loss)), lbfgs_reg_loss: $(median(data_df.lbfgs_reg_loss))"

fig1 = histogram(data_df[!, :init_reg_loss], bins=20, label="before regularization", xlabel="regularization loss", alpha=0.8)
histogram!(data_df[!, :lbfgs_reg_loss], bins=20, label="after regularization", xlabel="regularization loss", alpha=0.8)

fig2 = histogram(filter(x -> x < 1, data_df[!, :adam_train_loss]), bins=20, label="before regularization", xlabel="train loss", alpha=0.8)
histogram!(filter(x -> x < 1, data_df[!, :lbfgs_train_loss]), bins=20, label="after regularization", xlabel="train loss", alpha=0.8)

fig3 = histogram(filter(x -> x < 1, data_df[!, :adam_val_loss]), bins=20, label="before regularization", xlabel="val loss", alpha=0.8)
histogram!(filter(x -> x < 1, data_df[!, :lbfgs_val_loss]), bins=20, label="after regularization", xlabel="val loss", alpha=0.8)

fig = plot(fig1, fig2, fig3, layout=(1, 3), size=(900, 300), fontfamily="Times", bottom_margin=5Plots.mm, left_margin=5Plots.mm, dpi=300)
# savefig(fig, "src/v2/plots/figures/k50_reg($reg_gamma)_loss_hist.png")
