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

struct ParamEstimator{L, C} <: Lux.AbstractLuxContainerLayer{(:lstm_model, :classifier)}
	lstm_model::L
	classifier::C
end

function (s::ParamEstimator)(x::AbstractArray{T, 2}, ps::Union{ComponentVector, NamedTuple}, st::NamedTuple) where {T}
	y, st_lstm = s.lstm_model(x, ps.lstm_model, st.lstm_model)
    y_mat = reduce(hcat, y)
	classifier_y, st_classifier = s.classifier(y_mat, ps.classifier, st.classifier)
	st = merge(st, (classifier = st_classifier, lstm_model = st_lstm))
	return classifier_y, st
end

function build_param_estimator(hidd_dims = 16)
	lstm_model = Recurrence(LSTMCell(3 => hidd_dims), return_sequence=true)
	classifier = Dense(hidd_dims => 2, sigmoid)
	return ParamEstimator(lstm_model, classifier)
end

function get_default_params(; hidd_dims = 8, rng = StableRNG(42))
	param_estimator = build_param_estimator(hidd_dims)
	ps_param_estimator, _ = Lux.setup(rng, param_estimator)
	return ComponentVector(
		ps = ps_param_estimator,
	)
end

function build_DPL_HBV(itpfuncs;
	hidd_dims = 8, rng = StableRNG(42),
	hbv_initstates = [0.01, 0.01, 0.01, 0.01, 0.01],
)
	itp_Lday, itp_P, itp_T = itpfuncs
	param_estimator = build_param_estimator(hidd_dims)
	ps_param_estimator, ps_param_st = Lux.setup(rng, param_estimator)
	psnn_func(x, p) = param_estimator(x, p, ps_param_st)[1]

	function solve_prob(initstates, params, timesteps, time_varying_params)
		@views BETA_vec, GAMMA_vec = time_varying_params[1,:], time_varying_params[2,:]

		function DPL_HBV_ODE_core(S, p, t)
            TT, CFMAX, CFR, CWH, LP, FC, _, PPERC, UZL, k0, k1, k2 = view(p, :hbv_ps)

            Lday = itp_Lday(t)
            P    = itp_P(t)
            T    = itp_T(t)
            # unnormalize
            BETA, GAMMA = BETA_vec[Int(t)] * 5.0 + 1.0, GAMMA_vec[Int(t)] * 5.0 + 1.0

            # snowpack, meltwater, soilwater, upper zone, lower zone
            @views S1, S2, S3, S4, S5 = S[1], S[2], S[3], S[4], S[5]

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

		prob = DiscreteProblem(
			DPL_HBV_ODE_core,
			initstates,
			Float64.((timesteps[1], timesteps[end])),
			params,
		)
		sol = solve(
			prob,
			FunctionMap{false}(),
		)
		Array(sol)
	end

	function DPL_HBV_model(input, params, timesteps)
		TT, CFMAX, CFR, CWH, LP, FC, _, PPERC, UZL, k0, k1, k2 = view(params, :hbv_ps)
		time_varying_params = psnn_func(input, params[:lstm_ps])
		sol_arr = solve_prob(hbv_initstates, params, timesteps, time_varying_params)
		s4_vec, s5_vec = view(sol_arr, 4, :), view(sol_arr, 5, :)
		q0_vec = Q0.(s4_vec, Ref(k0), Ref(UZL))
		q1_vec = Q1.(s4_vec, Ref(k1))
		q2_vec = Q2.(s5_vec, Ref(k2))

		Q_out_vec = q0_vec .+ q1_vec .+ q2_vec
		return Q_out_vec 
	end

	return DPL_HBV_model
end

