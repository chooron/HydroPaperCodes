# exphydro pro with ENN

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

# OPTq formula 1
Q1(S1, melt, rainfall, f, f1, fp, Smax) = ifelse(
	S1 + fp * (melt + rainfall) > Smax,
	f * max(0, Smax - fp * (melt + rainfall)) + f1 * max(0, S1 + fp * (melt + rainfall) - Smax),
	f * S1,
)
# OPTq formula 2
Q2(S1, melt, rainfall, f, f1, fp, Smax, Qmax) = ifelse(
	S1 + fp * (melt + rainfall) > Smax,
	Qmax + f1 * max(0, S1 + fp * (melt + rainfall) - Smax),
	Qmax * exp(-f * max(0, Smax - S1)),
)


function build_exphydro_pro_1(itps; solve_type = "discrete")
	itp_Lday, itp_P, itp_T = itps

	function epxhydro_ode_core!(dS, S, ps, t)
		f, Smax, Df, Tmax, Tmin, f1, fp = ps

		Lday = itp_Lday(t)
		P    = itp_P(t)
		T    = itp_T(t)

		melt = M(S[1], T, Df, Tmax)
		rainfall = Pr(P, T, Tmin)
		snowfall = Ps(P, T, Tmin)
		Q_out = Q1(S[2], melt, rainfall, f, f1, fp, Smax)

		dS[1] = snowfall - melt
		dS[2] = rainfall + melt - ET(S[2], T, Lday, Smax) - Q_out
	end

	function exphydro_model(input, params, initstates, t_out; return_all = false)
		f, Smax, Df, Tmax, Tmin, f1, fp = params
		if solve_type == "discrete"
			prob = DiscreteProblem(epxhydro_ode_core!, initstates, Float64.((t_out[1], maximum(t_out))))
			sol = solve(prob, FunctionMap{true}(), u0 = initstates, p = params, sensealg = ForwardDiffSensitivity())
		else
			prob = ODEProblem(epxhydro_ode_core!, initstates, Float64.((t_out[1], maximum(t_out))))
			sol = solve(prob, BS3(), u0 = initstates, p = params, saveat = t_out, reltol = 1e-2, abstol = 1e-2, sensealg = ForwardDiffSensitivity())
		end

		prcp_, temp_, lday_ = input[1, :], input[2, :], input[3, :]
		melt_ = M.(sol[1, :], temp_, Ref(Df), Ref(Tmax))
		ps_ = Ps.(prcp_, temp_, Ref(Tmin))
		pr_ = Pr.(prcp_, temp_, Ref(Tmin))
		et_ = ET.(sol[2, :], temp_, lday_, Ref(Smax))
		Qout_ = Q1.(sol[2, :], melt_, pr_, Ref(f), Ref(f1), Ref(fp), Ref(Smax))
		return return_all ? (snowpack = sol[1, :], soilwater = sol[2, :], pr = pr_, ps = ps_, melt = melt_, et = et_, qsim = Qout_) : Qout_
	end
end

function build_exphydro_pro_2(itps; solve_type = "discrete")
	itp_Lday, itp_P, itp_T = itps

	function epxhydro_ode_core!(dS, S, ps, t)
		f, Smax, Qmax, Df, Tmax, Tmin, f1, fp = ps

		Lday = itp_Lday(t)
		P    = itp_P(t)
		T    = itp_T(t)

		snowfall = Ps(P, T, Tmin)
		rainfall = Pr(P, T, Tmin)
		melt = M(S[1], T, Df, Tmax)

		Q_out = Q2(S[2], melt, rainfall, f, f1, fp, Smax, Qmax)

		dS[1] = snowfall - melt
		dS[2] = rainfall + melt - ET(S[2], T, Lday, Smax) - Q_out
	end

	function exphydro_model(input, params, initstates, t_out; return_all = false)
		f, Smax, Qmax, Df, Tmax, Tmin, f1, fp = params
		if solve_type == "discrete"
			prob = DiscreteProblem(epxhydro_ode_core!, initstates, Float64.((t_out[1], maximum(t_out))))
			sol = solve(prob, FunctionMap{true}(), u0 = initstates, p = params, sensealg = ForwardDiffSensitivity())
		else
			prob = ODEProblem(epxhydro_ode_core!, initstates, Float64.((t_out[1], maximum(t_out))))
			sol = solve(prob, BS3(), u0 = initstates, p = params, saveat = t_out, reltol = 1e-2, abstol = 1e-2, sensealg = ForwardDiffSensitivity())
		end

		prcp_, temp_, lday_ = input[1, :], input[2, :], input[3, :]
		melt_ = M.(sol[1, :], temp_, Ref(Df), Ref(Tmax))
		ps_ = Ps.(prcp_, temp_, Ref(Tmin))
		pr_ = Pr.(prcp_, temp_, Ref(Tmin))
		et_ = ET.(sol[2, :], temp_, lday_, Ref(Smax))
		Qout_ = Q2.(sol[2, :], melt_, pr_, Ref(f), Ref(f1), Ref(fp), Ref(Smax), Ref(Qmax))
		return return_all ? (snowpack = sol[1, :], soilwater = sol[2, :], pr = pr_, ps = ps_, melt = melt_, et = et_, qsim = Qout_) : Qout_
	end
end

