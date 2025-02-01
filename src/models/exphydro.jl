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

function build_exphydro(itps; solve_type = "discrete")
	itp_Lday, itp_P, itp_T = itps

	function epxhydro_ode_core!(dS, S, ps, t)
		f, Smax, Qmax, Df, Tmax, Tmin = ps

		Lday = itp_Lday(t)
		P    = itp_P(t)
		T    = itp_T(t)

		Q_out = Qb(S[2], f, Smax, Qmax) + Qs(S[2], Smax)

		dS[1] = Ps(P, T, Tmin) - M(S[1], T, Df, Tmax)
		dS[2] = Pr(P, T, Tmin) + M(S[1], T, Df, Tmax) - ET(S[2], T, Lday, Smax) - Q_out
	end

	function exphydro_model(input, params, initstates, t_out; return_all = false)
		f, Smax, Qmax, Df, Tmax, Tmin = params
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
		Qb_ = Qb.(sol[2, :], Ref(f), Ref(Smax), Ref(Qmax))
		Qs_ = Qs.(sol[2, :], Ref(Smax))
		Qout_ = Qb_ .+ Qs_
		return return_all ? (snowpack = sol[1, :], soilwater = sol[2, :], pr = pr_, ps = ps_, melt = melt_, et = et_, qsim = Qout_) : Qout_
	end
end

function build_exphydrov2(itps; solve_type = "discrete")
	itp_Lday, itp_P, itp_T = itps

	function epxhydro_ode_core!(dS, S, ps, t)
		f, Smax, Qmax, Df, Tmax, Tmin, pinfil = ps

		Lday = itp_Lday(t)
		P    = itp_P(t)
		T    = itp_T(t)

		snowfall = Ps(P, T, Tmin)
		rainfall = Pr(P, T, Tmin)
		melt = M(S[1], T, Df, Tmax)

		infil = (rainfall + melt) * pinfil
		surface_flow = rainfall + melt - infil

		Q_out = Qb(S[2], f, Smax, Qmax) + Qs(S[2], Smax) + surface_flow

		dS[1] = snowfall - melt
		dS[2] = infil - ET(S[2], T, Lday, Smax) - Q_out
	end

	function exphydro_model(input, params, initstates, t_out; return_all = false)
		f, Smax, Qmax, Df, Tmax, Tmin, pinfil = params
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
		infil_ = (melt_ .+ pr_) .* pinfil
		surface_flow_ = pr_ .+ melt_ - infil_
		et_ = ET.(sol[2, :], temp_, lday_, Ref(Smax))
		Qb_ = Qb.(sol[2, :], Ref(f), Ref(Smax), Ref(Qmax))
		Qs_ = Qs.(sol[2, :], Ref(Smax))
		Qout_ = Qb_ .+ Qs_ .+ surface_flow_
		return return_all ? (snowpack = sol[1, :], soilwater = sol[2, :], pr = pr_, ps = ps_, melt = melt_, et = et_, qsim = Qout_) : Qout_
	end
end