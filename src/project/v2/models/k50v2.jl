#* 这个是一个改进的ExpHydro的神经网络耦合版本
#* 主要改进有以下几点
#* 1. 将融雪数据也作为模型的输入
#* 2. 取消原有的数据转换,比如Exp,Log,Normalize等,直接使用神经网络对数据进行预测
#* 3. 使用神经网络分别构建两个神经网络,一个用于预测ET,一个用于预测Q, ETNN以(snowpack, soilwater, temp)为输入, QNN以(snowpack, soilwater, rainfall)为输入

using Lux
using KolmogorovArnold
include("MultiKAN.jl")

function build_ET_NN(hidd_dims=4, grid_size=4, basis_func=rbf)
    normalizer = tanh_fast
    layer_type = KDense
    width = [3, hidd_dims, 1]
    mult_arity = [Dict(1 => 2, 2 => 2)]
    layer_cfg = [
        Dict(:grid_len => grid_size, :use_base_act => true, :normalizer => normalizer, :basis_func => basis_func),
        Dict(:grid_len => grid_size, :use_base_act => true, :normalizer => normalizer, :basis_func => basis_func)
    ]
    etnn = KAN(width, mult_arity, layer_cfg, layer_type)
    return etnn
end

function build_Q_NN(hidd_dims=4, grid_size=4, basis_func=rbf)
    normalizer = tanh_fast
    layer_type = KDense
    width = [3, hidd_dims, 1]
    mult_arity = [Dict(1 => 2, 2 => 2)]
    layer_cfg = [
        Dict(:grid_len => grid_size, :use_base_act => true, :normalizer => normalizer, :basis_func => basis_func),
        Dict(:grid_len => grid_size, :use_base_act => true, :normalizer => normalizer, :basis_func => basis_func)
    ]
    qnn = KAN(width, mult_arity, layer_cfg, layer_type)
    return qnn
end

function build_K50_NNs(hidd_dims=4, grid_size=4, basis_func=rbf)
    #* input is norm_snowpack, norm_soilwater, norm_temp, output is evap
    etnn = build_ET_NN(hidd_dims, grid_size, basis_func)
    #* input is norm_soilwater, norm_prcp
    qnn = build_Q_NN(hidd_dims, grid_size, basis_func)
    return etnn, qnn
end

function build_Model_F(itpfuncs, normfuncs, nnfuncs; initstates=[1e-3, 1e-3], solve_type="discrete")
    p_itp, t_itp, l_itp = itpfuncs
    norm_s0_func, norm_s1_func, norm_temp_func, norm_infil_func = normfuncs
    etnn_func, qnn_func = nnfuncs

    step_fct(x) = (tanh(5.0 * x) + 1.0) * 0.5
    # snow precipitation
    Ps(P, T, Tmin) = step_fct(Tmin - T) * P
    # rain precipitation
    Pr(P, T, Tmin) = step_fct(T - Tmin) * P
    # snow melt
    M(S0, T, Df, Tmax) = step_fct(T - Tmax) * step_fct(S0) * minimum([S0, Df * (T - Tmax)])

    function ODE_core!(dS, S, p, t)
        @views Df, Tmax, Tmin = p[:exphydro][1], p[:exphydro][2], p[:exphydro][3]

        Lday, P, T = l_itp(t), p_itp(t), t_itp(t)

        @views S0, S1 = S[1], S[2]

        snowfall = Ps(P, T, Tmin)
        rainfall = Pr(P, T, Tmin)
        melting = M(S0, T, Df, Tmax)

        norm_s0, norm_s1, norm_temp = norm_s0_func(S0), norm_s1_func(S1), norm_temp_func(T)
        norm_infil = norm_infil_func(rainfall + melting)

        g_ET = etnn_func([norm_s0 norm_s1 norm_temp]', view(p, :et))
        g_Q = qnn_func([norm_s0 norm_s1 norm_infil]', view(p, :q))

        dS[1] = max(-S0, snowfall - melting)
        dS[2] = max(-S1, rainfall + melting - g_ET[1] * Lday - g_Q[1])
    end

    function solve_ode_prob(params, timesteps)
        prob = ODEProblem(ODE_core!, initstates, (timesteps[1], timesteps[end]), params)
        sol = solve(prob, BS3(), saveat=1.0, reltol=1e-2, abstol=1e-2, sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()))
        sol
    end

    function solve_discrete_prob(params, timesteps)
        prob = DiscreteProblem(ODE_core!, initstates, (timesteps[1], timesteps[end]), params)
        sol = solve(prob, FunctionMap{true}(), saveat=1.0)
        sol
    end

    solve_prob = solve_type == "discrete" ? solve_discrete_prob : solve_ode_prob

    function Model_F_Core(input, params, timesteps)
        Df, Tmax, Tmin = params[:exphydro][1], params[:exphydro][2], params[:exphydro][3]
        sol = solve_prob(params, timesteps)
        s0_vec, s1_vec = sol[1, :], sol[2, :]
        norm_s0_vec, norm_s1_vec = norm_s0_func.(s0_vec), norm_s1_func.(s1_vec)
        rainfall_vec = Pr.(view(input, 1, :), view(input, 2, :), Ref(Tmin))
        melting_vec = M.(s0_vec, view(input, 2, :), Ref(Df), Ref(Tmax))
        norm_infil_vec = norm_infil_func.(rainfall_vec .+ melting_vec)
        Qout_ = view(qnn_func(permutedims([norm_s0_vec norm_s1_vec norm_infil_vec]), params[:q]), 1, :)
        # (q = Qout_, s0 = s0_vec, s1 = s1_vec, infil = infil_vec, melt = melting_vec, rainfall = rainfall_vec)
        return vec(Qout_)
    end

    return Model_F_Core
end

function build_Symbolize_F(itpfuncs, normfuncs, qnn_func; initstates=[1e-3, 1e-3], solve_type="discrete")
    p_itp, t_itp, l_itp = itpfuncs
    norm_s0_func, norm_s1_func, norm_infil_func = normfuncs
    # # evapotranspiration
    PET(T, Lday) = 29.8 * Lday * 0.611 * exp((17.3 * T) / (T + 237.3)) / (T + 273.2)
    ET(S1, T, Lday, Smax) = PET(T, Lday) * min(1.0, S1 / Smax)
    step_fct(x) = (tanh(5.0 * x) + 1.0) * 0.5
    # snow precipitation
    Ps(P, T, Tmin) = step_fct(Tmin - T) * P
    # rain precipitation
    Pr(P, T, Tmin) = step_fct(T - Tmin) * P
    # snow melt
    M(S0, T, Df, Tmax) = step_fct(T - Tmax) * step_fct(S0) * minimum([S0, Df * (T - Tmax)])

    function ODE_core!(dS, S, p, t)
        @views Smax, Df, Tmax, Tmin = p[:exphydro][1], p[:exphydro][2], p[:exphydro][3], p[:exphydro][4]

        Lday, P, T = l_itp(t), p_itp(t), t_itp(t)

        @views S0, S1 = S[1], S[2]

        snowfall = Ps(P, T, Tmin)
        rainfall = Pr(P, T, Tmin)
        melting = M(S0, T, Df, Tmax)

        norm_s0, norm_s1 = norm_s0_func(S0), norm_s1_func(S1)
        norm_infil = norm_infil_func(rainfall + melting)

        g_Q = qnn_func([norm_s0, norm_s1, norm_infil], view(p, :q))

        # dS[1] = max(-S0, snowfall - melting)
        # dS[2] = max(-S1, rainfall + melting - ET(S1, T, Lday, Smax) - g_Q)

        dS[1] = snowfall - melting
        dS[2] = rainfall + melting - ET(S1, T, Lday, Smax) - g_Q
    end

    function solve_ode_prob(params, timesteps)
        prob = ODEProblem(ODE_core!, initstates, Float64.((timesteps[1], timesteps[end])), params)
        sol = solve(prob, BS3(), saveat=1.0, reltol=1e-2, abstol=1e-2, sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()))
        Array(sol)
    end

    function solve_discrete_prob(params, timesteps)
        prob = DiscreteProblem(ODE_core!, initstates, Float64.((timesteps[1], timesteps[end])), params)
        sol = solve(prob, FunctionMap{true}(), saveat=1.0)
        Array(sol)
    end

    solve_prob = solve_type == "discrete" ? solve_discrete_prob : solve_ode_prob

    function Model_F_Core(input, params, timesteps)
        sol_arr = solve_prob(params, timesteps)
        Df, Tmax, Tmin = params[:exphydro][2], params[:exphydro][3], params[:exphydro][4]
        s0_vec, s1_vec = view(sol_arr, 1, :), view(sol_arr, 2, :)
        norm_s0_vec, norm_s1_vec = norm_s0_func.(s0_vec), norm_s1_func.(s1_vec)
        rainfall_vec = Pr.(view(input, 1, :), view(input, 2, :), Ref(Tmin))
        melting_vec = M.(s0_vec, view(input, 3, :), Ref(Df), Ref(Tmax))
        norm_infil_vec = norm_infil_func.(rainfall_vec .+ melting_vec)
        qnn_input = permutedims([norm_s0_vec norm_s1_vec norm_infil_vec])
        qnn_func_output = reduce(hcat, qnn_func.(eachslice(qnn_input, dims=2), Ref(params[:q])))
        Qout_ = view(qnn_func_output, 1, :)[:, 1]
        return Qout_
    end

    return Model_F_Core
end
