#if !defined(SINGLE_BAND_HALF_FILLING_DMFT_IPT_SOLVER)
#define SINGLE_BAND_HALF_FILLING_DMFT_IPT_SOLVER

#include "green.h"

// Half-filling IPT solver for DMFT
class IPTSolver
{
public:
	IPTSolver(const double U, const double BETA, const size_t Nt, const size_t Nw)
	: _U(U), _BETA(BETA), _Nt(Nt), _Nw(Nw), _G_iw_history(Nw) {}

	double solve(MatsubaraFreqGreen& G_iw, const MatsubaraFreqGreen& g0_iw, const std::vector<double> M)
	{
		MatsubaraTimeGreen g0_tau = fft_InverseFourier(&g0_iw[0],_Nw,_BETA,M,_Nt);
		MatsubaraTimeGreen sigma_tau = std::pow(_U,2) * g0_tau * g0_tau * g0_tau;
		MatsubaraFreqGreen sigma_iw = fft_Fourier(&sigma_tau[0],_Nt,_BETA,_Nw);

		G_iw = inverse(inverse(g0_iw) - sigma_iw);

		double distance = dist(G_iw,_G_iw_history);

		_G_iw_history = G_iw;

		return distance;
	}

private:
	double dist(MatsubaraFreqGreen& g1, MatsubaraFreqGreen& g2) const
	{
		double accum = 0;
		for(int i=0;i<_Nw;++i)
			accum += std::norm(g1[i] - g2[i]);
		return accum;
	}

	const double _U;
	const double _BETA;
	const size_t _Nt;
	const size_t _Nw;

	MatsubaraFreqGreen _G_iw_history;
};

#endif
