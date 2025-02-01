using Lux
using KolmogorovArnold

# snowpack
Rainfall(prcp, temp, TT) = ifelse(temp - TT > 0, prcp, 0.0)
Snowfall(prcp, temp, TT) = ifelse(TT - temp > 0, prcp, 0.0)
Melt(temp, snowpack, TT, CFMAX) = min(snowpack, max(0.0, temp - TT) * CFMAX)
Refreeze(temp, meltwater, TT, CFR, CFMAX) = min(max((TT - temp), 0.0) * CFR * CFMAX, meltwater)
Infiltration(snowpack, meltwater, CWH) = max(meltwater - CWH * snowpack, 0.0)
# soilwater
Recharge(soilwater, rainfall, infil, FC, BETA) = (rainfall + infil) * clamp(max(0.0, soilwater / FC)^BETA, 0, 1)
Excess(soilwater, FC) = max(soilwater - FC, 0.0)
PET(temp, lday) = 29.8 * lday * 0.611 * exp((17.3 * temp) / (temp + 237.3)) / (temp + 273.2)
# add GAMMA
Evap(soilwater, temp, lday, GAMMA, LP, FC) = min(clamp((max(0.0, soilwater / (LP * FC))) ^ GAMMA, 0, 1) * PET(temp, lday), soilwater)
# freewater
Percolation(suz, PPERC) = min(suz, PPERC)
Q0(suz, k0, UZL) = max(0.0, suz - UZL) * k0
Q1(suz, k1) = suz * k1
Q2(slz, k2) = slz * k2


function build_K50_NNs(; hidd_dims = 4, grid_len = 4)
	#* input is norm_snowpack, norm_soilwater, norm_temp, norm_prcp, output is Smax and Qmax parameters
	basis_func = rbf
	normalizer = tanh_fast
	psnn = Chain(
		KDense(2, hidd_dims, grid_len; use_base_act = true, basis_func, normalizer),
		KDense(hidd_dims, 2, grid_len; use_base_act = true, basis_func, normalizer),
		x -> sigmoid_fast.(x),
	)
	return psnn
end

function get_default_params(; hidd_dims = 4, grid_len = 4, rng = StableRNG(42))
	ann_ps = build_K50_NNs(hidd_dims = hidd_dims, grid_len = grid_len)
	ps_nn_ps, _ = Lux.setup(rng, ann_ps)
	return ComponentVector(
		ps = ps_nn_ps,
	)
end

function build_K50_P(itpfuncs, normfuncs;
	hidd_dims = 4, grid_len = 4, rng = StableRNG(42),
	hbv_initstates = [0.01, 0.01, 0.01, 0.01, 0.01],
)
	itp_Lday, itp_P, itp_T = itpfuncs
	norm_S2, norm_S3 = normfuncs
	ann_ps = build_K50_NNs(hidd_dims = hidd_dims, grid_len = grid_len)
	ps_nn_ps, ps_nn_st = Lux.setup(rng, ann_ps)
	psnn_func(x, p) = ann_ps(x, p, ps_nn_st)[1]

	function K50_ODE_core!(S, p, t)
		TT, CFMAX, CFR, CWH, LP, FC, _, PPERC, UZL, k0, k1, k2 = view(p, :hbv_ps)

		Lday = itp_Lday(t)
		P    = itp_P(t)
		T    = itp_T(t)

		# snowpack, meltwater, soilwater, upper zone, lower zone
		@views S1, S2, S3, S4, S5 = S[1], S[2], S[3], S[4], S[5]
		g_ps = psnn_func([norm_S2(S2), norm_S3(S3)], view(p, :kan_ps))
		# unnormalize
		@views BETA, GAMMA = g_ps[1] * 5.0 + 1.0, g_ps[2] * 5.0 + 1.0
		
		# Snowpack
		rainfall = Rainfall(P, T, TT)
		snowfall = Snowfall(P, T, TT)
		S1 = S1 + snowfall

		# Meltwater		
		melt = Melt(T, S1, TT, CFMAX)
		S1 = S1 - melt
		S2 = S2 + melt

		# Refreeze
		refreeze = Refreeze(T, S2, TT, CFR, CFMAX)
		S1 = S1 + refreeze
		S2 = S2 - refreeze

		# Infiltration
		infil = clamp(Infiltration(S1, S2, CWH), 0.0, S2)
		S2 = S2 - infil

		# Recharge
		recharge = Recharge(S3, rainfall, infil, FC, BETA)
		S3 = S3 + rainfall + infil - recharge

		# Excess
		excess = Excess(S3, FC)
		S3 = S3 - excess

		# Evap
		evap = Evap(S3, T, Lday, GAMMA, LP, FC)
		S3 = max(S3 - evap, 1e-6)

		# Percolation
		S4 = S4 + recharge + excess
		percolation = Percolation(S4, PPERC)
		S4 = S4 - percolation
		S5 = S5 + percolation

		# Upper zone
		q0 = Q0(S4, k0, UZL)
		S4 = S4 - q0
		q1 = Q1(S4, k1)
		S4 = S4 - q1
		q2 = Q2(S5, k2)
		S5 = S5 - q2

		# assert non-negative
		[S1, S2, S3, S4, S5]
	end

	function solve_prob(initstates, params, timesteps)
		prob = DiscreteProblem(
			K50_ODE_core!,
			initstates,
			Float64.((timesteps[1], timesteps[end])),
			params,
		)
		sol = solve(prob, FunctionMap{false}())
		Array(sol)
	end

	function K50_P_model(input, params, timesteps)
		TT, CFMAX, CFR, CWH, LP, FC, _, PPERC, UZL, k0, k1, k2 = view(params, :hbv_ps)
		sol_arr = solve_prob(hbv_initstates, params, timesteps)
		s4_vec, s5_vec = view(sol_arr, 4, :), view(sol_arr, 5, :)
		q0_vec = Q0.(s4_vec, Ref(k0), Ref(UZL))
		q1_vec = Q1.(s4_vec, Ref(k1))
		q2_vec = Q2.(s5_vec, Ref(k2))

		Q_out_vec = q0_vec .+ q1_vec .+ q2_vec
		return Q_out_vec 
	end

	return K50_P_model
end

