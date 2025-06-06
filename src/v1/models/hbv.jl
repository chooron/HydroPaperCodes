# functions for hbv model
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
Evap(soilwater, temp, lday, LP, FC) = min(clamp(max(0.0, soilwater / (LP * FC)), 0, 1) * PET(temp, lday), soilwater)
# freewater
Percolation(suz, PPERC) = min(suz, PPERC)
Q0(suz, k0, UZL) = max(0.0, suz - UZL) * k0
Q1(suz, k1) = suz * k1
Q2(slz, k2) = slz * k2


function build_c_hbv(itps)
	itp_Lday, itp_P, itp_T = itps

	function hbv_ode_core!(dS, S, ps, t)
		TT, CFMAX, CFR, CWH, LP, FC, BETA, PPERC, UZL, k0, k1, k2 = ps

		Lday = itp_Lday(t)
		P    = itp_P(t)
		T    = itp_T(t)

		# snowpack, meltwater, soilwater, upper zone, lower zone
		@views S1, S2, S3, S4, S5 = S[1], S[2], S[3], S[4], S[5]

		rainfall = Rainfall(P, T, TT)
		snowfall = Snowfall(P, T, TT)
		melt = Melt(T, S1, TT, CFMAX)
		refreeze = Refreeze(T, S2, TT, CFR, CFMAX)
		infil = clamp(Infiltration(S1, S2, CWH), 0.0, S2)

		recharge = Recharge(S3, rainfall, infil, FC, BETA)
		excess = Excess(S3, FC)
		evap = Evap(S3, T, Lday, LP, FC)

		percolation = Percolation(S4, PPERC)
		q0 = Q0(S4, k0, UZL)
		q1 = Q1(S4, k1)
		q2 = Q2(S5, k2)

		# assert non-negative
		dS[1] = max(-S1, snowfall + refreeze - melt)
		dS[2] = max(-S2, melt - refreeze - infil)
		dS[3] = max(-S3, rainfall + infil - recharge - excess - evap)
		dS[4] = max(-S4, recharge + excess - percolation - q0 - q1)
		dS[5] = max(-S5, percolation - q2)
	end

	function hbv_model(input, params, initstates, t_out; return_all = false)
		TT, CFMAX, CFR, CWH, LP, FC, BETA, PPERC, UZL, k0, k1, k2 = p_vec = Vector(params)
		prob = ODEProblem(hbv_ode_core!, initstates, Float64.((t_out[1], maximum(t_out))))
		sol = solve(prob, BS3(), u0 = initstates, p = p_vec, saveat = t_out, reltol = 1e-3, abstol = 1e-3, sensealg = ForwardDiffSensitivity())
		@views S1_, S2_, S3_, S4_, S5_ = sol[1, :], sol[2, :], sol[3, :], sol[4, :], sol[5, :]
		prcp_, temp_, lday_ = input[1, :], input[2, :], input[3, :]

		rainfall_ = Rainfall.(prcp_, temp_, Ref(TT))
		snowfall_ = Snowfall.(prcp_, temp_, Ref(TT))
		melt_ = Melt.(temp_, S1_, Ref(TT), Ref(CFMAX))
		refreeze_ = Refreeze.(temp_, S2_, Ref(TT), Ref(CFR), Ref(CFMAX))
		infil_ = Infiltration.(S1_, S2_, Ref(CWH))

		recharge_ = Recharge.(S3_, rainfall_, infil_, Ref(FC), Ref(BETA))
		excess_ = Excess.(S3_, Ref(FC))
		evap_ = Evap.(S3_, temp_, lday_, Ref(LP), Ref(FC))

		percolation_ = Percolation.(S4_, Ref(PPERC))
		q0_ = Q0.(S4_, Ref(k0), Ref(UZL))
		q1_ = Q1.(S4_, Ref(k1))
		q2_ = Q2.(S5_, Ref(k2))

		Qout_ = q0_ .+ q1_ .+ q2_

		if return_all
			return (
				rainfall = rainfall_, snowfall = snowfall_, melt = melt_, refreeze = refreeze_,
				infil = infil_, recharge = recharge_, excess = excess_,
				evap = evap_, percolation = percolation_,
				q0 = q0_, q1 = q1_, q2 = q2_, Qout = Qout_,
				S1 = S1_, S2 = S2_, S3 = S3_, S4 = S4_, S5 = S5_,
			)
		else
			return Qout_
		end
	end
end

function build_d_hbv(itps)
	itp_Lday, itp_P, itp_T = itps

	function hbv_ode_core!(S, ps, t)
		TT, CFMAX, CFR, CWH, LP, FC, BETA, PPERC, UZL, k0, k1, k2 = ps

		Lday = itp_Lday(t)
		P    = itp_P(t)
		T    = itp_T(t)

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
		evap = Evap(S3, T, Lday, LP, FC)
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

	function hbv_model(input, params, initstates, t_out; return_all = false)
		TT, CFMAX, CFR, CWH, LP, FC, BETA, PPERC, UZL, k0, k1, k2 = p_vec = Vector(params)
		prob = DiscreteProblem(hbv_ode_core!, initstates, Float64.((t_out[1], maximum(t_out))))
		sol = solve(prob, FunctionMap(), u0 = initstates, p = p_vec, saveat = t_out, sensealg = ForwardDiffSensitivity())
		@views S1_, S2_, S3_, S4_, S5_ = sol[1, :], sol[2, :], sol[3, :], sol[4, :], sol[5, :]
		prcp_, temp_, lday_ = input[1, :], input[2, :], input[3, :]

		rainfall_ = Rainfall.(prcp_, temp_, Ref(TT))
		snowfall_ = Snowfall.(prcp_, temp_, Ref(TT))
		melt_ = Melt.(temp_, S1_, Ref(TT), Ref(CFMAX))
		refreeze_ = Refreeze.(temp_, S2_, Ref(TT), Ref(CFR), Ref(CFMAX))
		infil_ = Infiltration.(S1_, S2_, Ref(CWH))

		recharge_ = Recharge.(S3_, rainfall_, infil_, Ref(FC), Ref(BETA))
		excess_ = Excess.(S3_, Ref(FC))
		evap_ = Evap.(S3_, temp_, lday_, Ref(LP), Ref(FC))

		percolation_ = Percolation.(S4_, Ref(PPERC))
		q0_ = Q0.(S4_, Ref(k0), Ref(UZL))
		q1_ = Q1.(S4_, Ref(k1))
		q2_ = Q2.(S5_, Ref(k2))

		Qout_ = q0_ .+ q1_ .+ q2_

		if return_all
			return (
				rainfall = rainfall_, snowfall = snowfall_, melt = melt_, refreeze = refreeze_,
				infil = infil_, recharge = recharge_, excess = excess_,
				evap = evap_, percolation = percolation_,
				q0 = q0_, q1 = q1_, q2 = q2_, Qout = Qout_,
				S1 = S1_, S2 = S2_, S3 = S3_, S4 = S4_, S5 = S5_,
			)
		else
			return Qout_
		end
	end
end

function build_d_hbv_without_zone(itps)
	itp_Lday, itp_P, itp_T = itps

	function hbv_ode_core!(S, ps, t)
		TT, CFMAX, CFR, CWH, LP, FC, BETA = ps

		Lday = itp_Lday(t)
		P    = itp_P(t)
		T    = itp_T(t)

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
		evap = Evap(S3, T, Lday, LP, FC)
		S3 = max(S3 - evap, 1e-6)

		# assert non-negative
		[S1, S2, S3]
	end

	function hbv_model(input, params, initstates, t_out; return_all = false)
		TT, CFMAX, CFR, CWH, LP, FC, BETA = p_vec = Vector(params)
		prob = DiscreteProblem(hbv_ode_core!, initstates, Float64.((t_out[1], maximum(t_out))))
		sol = solve(prob, FunctionMap(), u0 = initstates, p = p_vec, saveat = t_out, sensealg = ForwardDiffSensitivity())
		@views S1_, S2_, S3_, S4_, S5_ = sol[1, :], sol[2, :], sol[3, :], sol[4, :], sol[5, :]
		prcp_, temp_, lday_ = input[1, :], input[2, :], input[3, :]

		rainfall_ = Rainfall.(prcp_, temp_, Ref(TT))
		snowfall_ = Snowfall.(prcp_, temp_, Ref(TT))
		melt_ = Melt.(temp_, S1_, Ref(TT), Ref(CFMAX))
		refreeze_ = Refreeze.(temp_, S2_, Ref(TT), Ref(CFR), Ref(CFMAX))
		infil_ = Infiltration.(S1_, S2_, Ref(CWH))

		recharge_ = Recharge.(S3_, rainfall_, infil_, Ref(FC), Ref(BETA))
		excess_ = Excess.(S3_, Ref(FC))
		evap_ = Evap.(S3_, temp_, lday_, Ref(LP), Ref(FC))

		Qout_ = excess_ .+ excess_

		if return_all
			return (
				rainfall = rainfall_, snowfall = snowfall_, melt = melt_, refreeze = refreeze_,
				infil = infil_, recharge = recharge_, excess = excess_,	evap = evap_,
				Qout = Qout_, S1 = S1_, S2 = S2_, S3 = S3_,
			)
		else
			return Qout_
		end
	end
end