#include "green.h"
#include "pade.h"

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


int main(int argc,char* argv[])
{
	const double BETA = 30;
	const double U = std::atof(argv[1]);
	const size_t Nt = 10000;
	const size_t Nw = 1000;
	const double CONV = 1e-5;
	const size_t PADE_PRECISION = 520;
	const size_t PADE_N_POINTS = 300;
	std::vector<double> M = {1,0,0.25}; // spectral moments for the non-interacting Bethe-lattice green function.
	int numIter = 1;

	MatsubaraFreqGreen iOmega_n(Nw); iOmega_n.set_iOmega_n(BETA);
	MatsubaraFreqGreen g0_iw(Nw);
	MatsubaraFreqGreen G_iw(Nw);

	G_iw = create_BetheGf(BETA,0,Nw);

	IPTSolver IPT(U,BETA,Nt,Nw);

	// DMFT self-consistent loop
	while(true)
	{
		// Bath Green's function for Bethe lattice.
		g0_iw = inverse(iOmega_n - std::pow(1./2.,2)*G_iw); // band width = 1.

		double distance = IPT.solve(G_iw,g0_iw,M);

		double n = 1-density(G_iw,BETA,M);

		// From the perturbation theory [M. Potthoff et al. PRB 55, 16132 (1997)]
		M[1] = U*(n-1./2.); M[2] = 0.25 + std::pow(U*(n-1./2.),2) + std::pow(U,2)*n*(1-n);

		std::cout<<"iter: "<<numIter<<"    |G_init - G_fin| : "<<distance<<"    filling : "<<1-n<<std::endl<<std::flush;
		if(distance < CONV)
		{
			std::cout<<"converge!"<<std::endl;
			break;
		}
		numIter += 1;
	}	

	MatsubaraTimeGreen G_tau = fft_InverseFourier(&G_iw[0],Nw,BETA,M,Nt);

	G_tau.print("gtau.out",BETA);


	// analytic-continuation with the Pade approximation.
	if(argc >= 3)
	{
		std::cout<<"we are using the Pade approximation..."<<std::endl<<std::flush;
		std::ofstream file(argv[2]); // argv[2]: file name for storing a DOS.

		PadeForGreenFunction ancPade(PADE_PRECISION);
		ancPade.Fitting(&iOmega_n[0],&G_iw[0],PADE_N_POINTS);

		for(int i=0;i<PADE_N_POINTS;++i)
		{
			dcomplex w = -10 + i*20/(PADE_N_POINTS - 1.) + dcomplex(0,1e-3);
			dcomplex Aw = ancPade.estimate(w);
			file<<w.real()<<" "<<-1./M_PI*Aw.imag()<<std::endl;
		}
	}

	return 0;
}
