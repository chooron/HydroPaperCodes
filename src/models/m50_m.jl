function M50_ATTR()
	M50_INPUT = [:prcp, :temp, :lday]
	EXPHYDRO_PARAMS = [:f, :Smax, :Qmax, :Df, :Tmax, :Tmin]
	M50_PARAMS = [:exphydro, :et, :q, :est]
	return M50_INPUT, EXPHYDRO_PARAMS, M50_PARAMS
end

# smooting step function
step_fct(x) = (tanh(5.0 * x) + 1.0) * 0.5
# snow precipitation
Ps(P, T, Tmin) = step_fct(Tmin - T) * P
# rain precipitation
Pr(P, T, Tmin) = step_fct(T - Tmin) * P
# snow melt
M(S0, T, Df, Tmax) = step_fct(T - Tmax) * step_fct(S0) * minimum([S0, Df * (T - Tmax)])
# evapotranspiration
PET(T, Lday) = 29.8 * Lday * 0.611 * exp((17.3 * T) / (T + 237.3)) / (T + 273.2)
ET(S1, T, Lday, Smax) = step_fct(S1) * step_fct(S1 - Smax) * PET(T, Lday) + step_fct(S1) * step_fct(Smax - S1) * PET(T, Lday) * (S1 / Smax)
# base flow
Qb(S1, f, Smax, Qmax) = step_fct(S1) * step_fct(S1 - Smax) * Qmax + step_fct(S1) * step_fct(Smax - S1) * Qmax * exp(-f * (Smax - S1))
# peak flow
Qs(S1, Smax) = step_fct(S1) * step_fct(S1 - Smax) * (S1 - Smax)


function LSTMCompact(in_dims, hidden_dims, out_dims)
	lstm_cell = LSTMCell(in_dims => hidden_dims)
	classifier = Dense(hidden_dims => out_dims, sigmoid)
	return @compact(; lstm_cell, classifier) do x::AbstractArray{T, 3} where {T}
		x_init, x_rest = Iterators.peel(LuxOps.eachslice(x, Val(2)))
		y, carry = lstm_cell(x_init)
		for x in x_rest
			y, carry = lstm_cell((x, carry))
		end
		@return classifier(y)
	end
end

function build_initstate_estimator(hidd_dims = 16)
	lstm_model = LSTMCompact(3, 16, 2)
	return lstm_model
end

function build_M50_NNs(hidd_dims = 16)
	#* input is norm_snowpack, norm_soilwater, norm_temp, output is evap
	etnn = Lux.Chain(
		Lux.Dense(3, hidd_dims, tanh),
		Lux.Dense(hidd_dims, hidd_dims, leakyrelu),
		Lux.Dense(hidd_dims, 1, leakyrelu),
	)
	#* input is norm_soilwater, norm_prcp
	qnn = Lux.Chain(
		Lux.Dense(2, hidd_dims, tanh),
		Lux.Dense(hidd_dims, hidd_dims, leakyrelu),
		Lux.Dense(hidd_dims, 1, leakyrelu),
	)
	return etnn, qnn
end

function get_default_params_s(; hidd_dims = 16, rng = StableRNG(42))
	states_estimator = build_initstate_estimator(hidd_dims)
	ann_ET, ann_Q = build_M50_NNs(hidd_dims)
	et_ps, _ = Lux.setup(rng, ann_ET)
	q_ps, _ = Lux.setup(rng, ann_Q)
	est_ps, _ = Lux.setup(rng, states_estimator)
	return ComponentVector(
		exphydro = [0.01674478, 1709.461015, 18.46996175, 2.674548848, 0.175739196, -2.092959084],
		et = et_ps,
		q = q_ps,
	)
end

function get_default_params_m(; hidd_dims = 16, rng = StableRNG(42))
	states_estimator = build_initstate_estimator(hidd_dims)
	ann_ET, ann_Q = build_M50_NNs(hidd_dims)
	et_ps, _ = Lux.setup(rng, ann_ET)
	q_ps, _ = Lux.setup(rng, ann_Q)
	est_ps, _ = Lux.setup(rng, states_estimator)
	return ComponentVector(
		exphydro = [0.01674478, 1709.461015, 18.46996175, 2.674548848, 0.175739196, -2.092959084],
		et = et_ps,
		q = q_ps,
		est = est_ps,
	)
end

function build_M50(itpfuncs, normfuncs;
	m50_type = :s,
	hidd_dims = 16, rng = StableRNG(42),
	exphydro_params = ComponentVector(),
	exphydro_initstates = [],
	est_norms = [],
)
	p_itp, t_itp, l_itp = itpfuncs
	norm_S0, norm_S1, norm_T, norm_P = normfuncs
	states_estimator = build_initstate_estimator(hidd_dims)
	ann_ET, ann_Q = build_M50_NNs(hidd_dims)
	_, est_st = Lux.setup(rng, states_estimator)
	states_est_func(x, p) = Lux.apply(states_estimator, x, p, est_st)[1]
	etnn_func(x, p) = LuxCore.stateless_apply(ann_ET, x, p)
	qnn_func(x, p) = LuxCore.stateless_apply(ann_Q, x, p)

	function M50_ODE_core!(dS, S, p, t)
		@views Tmin, Tmax, Df = exphydro_params[6], exphydro_params[5], exphydro_params[4]

		Lday, P, T = l_itp(t), p_itp(t), t_itp(t)

		@views S0, S1 = S[1], S[2]
		norm_s0, norm_s1 = norm_S0(S0), norm_S1(S1)

		g_ET = etnn_func([norm_s0, norm_s1, norm_T(T)], view(p, :et))
		g_Q = qnn_func([norm_s1, norm_P(P)], view(p, :q))

		melting = M.(S0, T, Df, Tmax)

		dS[1] = Ps(P, T, Tmin) - melting
		dS[2] = Pr(P, T, Tmin) + melting - step_fct(S1) * Lday * exp(g_ET[1]) - step_fct(S1) * exp(g_Q[1])
	end

	function solve_prob(initstates, params, timesteps)
		prob = ODEProblem(
			M50_ODE_core!,
			initstates,
			Float64.((timesteps[1], timesteps[end])),
			params,
		)
		sol = solve(
			prob,
			BS3(),
			saveat = 1.0,
			reltol = 1e-3,
			abstol = 1e-3,
			sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP()),
		)
		Array(sol)
	end

	function solve_ensemble_prob(initstates, params, timesteps)
		trajectories = size(initstates, 2)
		prob = ODEProblem(
			M50_ODE_core!,
			[0.0, 1300.0],
			Float64.((timesteps[1, 1], maximum(timesteps[end, 1]))),
			params,
		)

		function prob_func(prob, i, repeat)
			remake(prob, u0 = [0.0, 1300.0], tspan = Float64.((timesteps[1, i], maximum(timesteps[:, i]))))
		end

		ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
		sol = solve(ensemble_prob, BS3(), EnsembleThreads(),
			saveat = 1.0,
			reltol = 1e-3,
			abstol = 1e-3,
			trajectories = trajectories,
			sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP()),
		)
		Array(sol)
	end

	function m50_model_s(input, params, timesteps)
		sol_arr = solve_prob(exphydro_initstates, params, timesteps)
		norm_prcp_vec = norm_P.(view(input, 1, :))
		norm_s1_vec = norm_S1.(view(sol_arr, 2, :))
		Qout_ = exp.(view(qnn_func(permutedims([norm_s1_vec norm_prcp_vec]), params[:q]), 1, :))
		return Qout_
	end

	function m50_model_ms(input, params, initstates, timesteps)
		# varnum, timesteps, batchsize
		sol_arr = solve_ensemble_prob(initstates, params, timesteps)
		@views prcp_arr, slw_arr = input[1, :, :], sol_arr[2, :, :]
		# timesteps, batchsize
		norm_prcp_mat = reduce(hcat, norm_P.(eachslice(prcp_arr, dims = 2)))
		norm_s1_mat = reduce(hcat, norm_S1.(eachslice(slw_arr, dims = 2)))
		input_mat = permutedims(cat(norm_prcp_mat, norm_s1_mat, dims = 3), (3, 1, 2))
		output_arr = reduce((c1, c2) -> cat(c1, c2, dims = 3), qnn_func.(eachslice(input_mat, dims = 3), Ref(params[:q])))
		exp.(view(output_arr, 1, :, :))
	end

	function m50_model_mh(input, params, timesteps; history_data = ones(3, 20, 10))
		s_init_prop = states_est_func(history_data, params[:est])
		s_init_pred = s_init_prop * (est_norms[2] .- est_norms[1]) .+ est_norms[1]
		sol_arr = solve_ensemble_prob(s_init_pred, params, timesteps)
		return sol_arr
	end

	if m50_type == :s
		return m50_model_s
	elseif m50_type == :ms
		return m50_model_ms
	elseif m50_type == :mh
		return m50_model_mh
	end
end
