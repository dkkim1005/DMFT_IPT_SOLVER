#include "green.h"
#include "pade.h"
#include "solver_ipt.h"
#include "argparser.h"

double doubleOccupancy(const MatsubaraFreqGreen& G_iw, const MatsubaraFreqGreen& self_energy_iw, const double U, const size_t Nt, const double BETA)
{
	auto kernel = self_energy_iw * G_iw;
	for(auto& k : kernel)
		k = std::conj(k);
	double D = 1./(U) * (fft_InverseFourier(&kernel[0],kernel.size(),BETA,{-U/2.,0,0},Nt))[0];

	return D;
}


int main(int argc,char* argv[])
{
	// cmd parser
	boost_argparser argparser(argc,argv);
	argparser.add_option("beta","inverse temperature");
	argparser.add_option("Uint","Coulomb interaction");
	argparser.add_option("padeFile","A name of a file storing a DOS");
	argparser.add_option("GFile","A name of a file storing a full green function");
	argparser.store();

	if(argc == 1)
	{
		argparser.load_help_option();
		return 1;
	}

	const double BETA = std::atof(argparser["beta"].c_str());
	const double U = std::atof(argparser["Uint"].c_str());
	const size_t Nt = 10001;
	const size_t Nw = 1000;
	const double CONV = 1e-4;
	const size_t PADE_PRECISION = 1024;
	const size_t PADE_N_POINTS = 220;
	std::vector<double> M = {1,0,0.25}; // spectral moments for the non-interacting Bethe-lattice green function.
	int numIter = 1;

	MatsubaraFreqGreen iOmega_n(Nw); iOmega_n.set_iOmega_n(BETA);
	MatsubaraFreqGreen g0_iw(Nw);
	MatsubaraFreqGreen G_iw(Nw);

	if(argparser.isOptionExist("GFile"))
		G_iw.read(argparser["GFile"].c_str());
	else
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

	std::string fileNameFullGreen = "G_iw" + std::to_string(U) + std::string(".out");
	G_iw.print(fileNameFullGreen.c_str(),BETA);


	// Double occupancy
	auto self_energy_iw = U/2. + inverse(g0_iw) - inverse(G_iw);

	double D = doubleOccupancy(G_iw, self_energy_iw, U, Nt, BETA);

	std::cout<<"Double occupancy:"<<D<<std::endl;


	// analytic-continuation with the Pade approximation.
	if(argparser.isOptionExist("padeFile"))
	{
		std::cout<<"we are using the Pade approximation..."<<std::endl<<std::flush;
		std::ofstream file(argparser["padeFile"].c_str()); // argv[2]: file name for storing a DOS.

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
