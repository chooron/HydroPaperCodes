include("../utils/criteria.jl")
using CSV, DataFrames, Dates

model_name = "k50_base"
base_path = "src/result/models/$model_name"
# Get all subdirectories in m50 folder
subdirs = filter(isdir, readdir(base_path, join = true))
criteria_all = []

# Read predict_df.csv from each subdir if it exists
for dir in subdirs
    train_pred_file = joinpath(dir, "train_predicted_df.csv")
    test_pred_file = joinpath(dir, "test_predicted_df.csv")
    station_id = basename(dir)
    train_df = CSV.read(train_pred_file, DataFrame)
    test_df = CSV.read(test_pred_file, DataFrame)

    # Split into train and test periods
    train_pred, train_obs = train_df[!, :pred], train_df[!, :obs]
    test_pred, test_obs = test_df[!, :val_pred], test_df[!, :obs]

    criteria_dict = Dict(
        "station_id" => station_id,
        "rmse-train" => rmse(train_obs, train_pred), "rmse-test" => rmse(test_obs, test_pred),
        "mae-train" => mae(train_obs, train_pred), "mae-test" => mae(test_obs, test_pred),
        "nse-train" => nse(train_obs, train_pred), "nse-test" => nse(test_obs, test_pred),
        "mnse-train" => mnse(train_obs, train_pred), "mnse-test" => mnse(test_obs, test_pred),
        "fhv-train" => fhv(train_obs, train_pred, h=0.01), "fhv-test" => fhv(test_obs, test_pred, h=0.01),
        "kge-train" => kge(train_obs, train_pred), "kge-test" => kge(test_obs, test_pred),
    )
    push!(criteria_all, criteria_dict)
end

criteria_df = DataFrame(criteria_all)
criteria_df_name = sort(filter(x -> x != "station_id", names(criteria_df)), rev=true)
criteria_df = criteria_df[!, ["station_id", criteria_df_name...]]

CSV.write("src/result/stats/$model_name-criteria.csv", criteria_df)


