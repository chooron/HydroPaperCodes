
using JLD2

function normalize_data(vec)
	mean, std = mean(vec), std(vec)
	norm_func = (x) -> (x .- mean) ./ std
	return norm_func.(vec), norm_func
end


function load_nn_data(basin_id, model_name, basemodel_name)
	#* load data
	camelsus_cache = load("src/data/camelsus/$(basin_id).jld2")
	data_x, data_y, data_timepoints = camelsus_cache["data_x"], camelsus_cache["data_y"], camelsus_cache["data_timepoints"]
    train_x, train_y, train_timepoints = camelsus_cache["train_x"], camelsus_cache["train_y"], camelsus_cache["train_timepoints"]
    lday_vec, prcp_vec, temp_vec = collect(data_x[:, 1]), collect(data_x[:, 2]), collect(data_x[:, 3])
    # Replace missing values with 0.0 in data arrays
    for data_arr in [data_x, data_y, train_x, train_y]
        replace!(data_arr, missing => 0.0)
    end

    #* load exphydro model outputs
    exphydro_df = CSV.read("src/result/models/$model_name/$(basin_id)/model_outputs.csv", DataFrame)
    snowpack_vec, soilwater_vec = exphydro_df[!, "snowpack"], exphydro_df[!, "soilwater"]
    et_vec, rainfall_vec, melt_vec = exphydro_df[!, "et"], exphydro_df[!, "pr"], exphydro_df[!, "melt"]
    et_vec[et_vec.<0] .= 0.000000001
    rainfall_vec[rainfall_vec.<0] .= 0.000000001
    melt_vec[melt_vec.<0] .= 0.000000001
    infil_vec = melt_vec .+ rainfall_vec

    #* normalize data
    s0_mean, s0_std = mean(snowpack_vec), std(snowpack_vec)
    s1_mean, s1_std = mean(soilwater_vec), std(soilwater_vec)
    t_mean, t_std = mean(temp_vec), std(temp_vec)
    infil_mean, infil_std = mean(infil_vec), std(infil_vec)

    norm_s0_func = (x) -> (x .- s0_mean) ./ s0_std
    norm_s1_func = (x) -> (x .- s1_mean) ./ s1_std
    norm_temp_func = (x) -> (x .- t_mean) ./ t_std
    norm_infil_func = (x) -> (x .- infil_mean) ./ infil_std
    normfuncs = (norm_s0_func, norm_s1_func, norm_temp_func, norm_infil_func)

    norm_snowpack = norm_s0_func.(snowpack_vec)
    norm_soilwater = norm_s1_func.(soilwater_vec)
    norm_temp = norm_temp_func.(temp_vec)
    norm_infil = norm_infil_func.(infil_vec)

    #* prepare training data
    nn_input = permutedims(reduce(hcat, (norm_snowpack, norm_soilwater, norm_temp, norm_infil)))

    if basemodel_name == "q"
        nn_input = nn_input[[1, 2, 4], 1:length(train_timepoints)]
    elseif basemodel_name == "et"
        nn_input = nn_input[[1, 2, 3], 1:length(train_timepoints)]
    else
        error("basemodel_name must be 'q' or 'et'")
    end
end