#* 表一
#* 主要是展示模型在正则化训练下与原模型的损失值的不同和模型参数reg_loss的值
using CSV, DataFrames, JLD2
include("../utils/train.jl")

basin_ids = ["02361000", "03281500", "06191500"]


reg_loss_info_vec = []
for basin_id in basin_ids
    model_dir = "src/result/models/k50_base(reg)/$(basin_id)"
    loss_df = CSV.read("$(model_dir)/loss_df.csv", DataFrame)
    min_train_loss_row = loss_df[argmin(loss_df.train_loss), :]
    push!(reg_loss_info_vec, min_train_loss_row)
end

reg_loss_info_df = DataFrame(reg_loss_info_vec)
reg_loss_info_df[!, :basin_id] = basin_ids

common_loss_info_vec = []
for basin_id in basin_ids
    model_dir = "src/result/models/k50_base/$(basin_id)"
    loss_df = CSV.read("$(model_dir)/loss_df.csv", DataFrame)
    min_train_loss_row = loss_df[argmin(loss_df.train_loss), :]
    tmp_opt_pas = load("$(model_dir)/train_records.jld2", "opt_ps")
    tmp_reg_loss = reg_loss(5e-2, [:et, :q])(tmp_opt_pas)
    min_train_loss_row.reg_loss = tmp_reg_loss
    push!(common_loss_info_vec, min_train_loss_row)
end
common_loss_info_df = DataFrame(common_loss_info_vec)
common_loss_info_df[!, :basin_id] = basin_ids
