#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <assert.h>
#include <fftw3.h>
#ifndef __GREENFUNCTION__
#define __GREENFUNCTION__
// compile option  : -std=c++11  or -std=c++14
// library linking : -lblas -llapack -lfftw3 

typedef std::complex<float> scomplex;
typedef std::complex<double> dcomplex;

extern "C" {
        void sgetrf_(int* m, int* n, float* a, int* lda, int* ipiv, int* info);
        void dgetrf_(int* m, int* n, double* a, int* lda, int* ipiv, int* info);
        void cgetrf_(int* m, int* n, void* a, int* lda, int* ipiv, int* info);
        void zgetrf_(int* m, int* n, void* a, int* lda, int* ipiv, int* info);

        void sgetri_(int* n, float* a, int* lda, int* ipiv,
                float* work, int* lwork, int* info);
        void dgetri_(int* n, double* a, int* lda, int* ipiv,
                double* work, int* lwork, int* info);
        void cgetri_(int* n, void* a, int* lda, int* ipiv,
                void* work, int* lwork, int* info);
        void zgetri_(int* n, void* a, int* lda, int* ipiv,
                void* work, int* lwork, int* info);
}

namespace lapack
{
        // Invert using getrf: general, s- d- c- z-
        void getri(int n, float* a, int* ipiv);
        void getri(int n, double* a, int* ipiv);
        void getri(int n, scomplex* a, int* ipiv);
        void getri(int n, dcomplex* a, int* ipiv);

        void getri(int n, float* a, int lda, int* ipiv);
        void getri(int n, double* a, int lda, int* ipiv);
        void getri(int n, scomplex* a, int lda, int* ipiv);
        void getri(int n, dcomplex* a, int lda, int* ipiv);


        // LU factorization : general, s- d- c- z-
        void getrf(int m, int n, float* a, int* ipiv);
        void getrf(int m, int n, double* a, int* ipiv);
        void getrf(int m, int n, scomplex* a, int* ipiv);
        void getrf(int m, int n, dcomplex* a, int* ipiv);

        void getrf(int m, int n, float* a, int lda, int* ipiv);
        void getrf(int m, int n, double* a, int lda,int* ipiv);
        void getrf(int m, int n, scomplex* a, int lda, int* ipiv);
        void getrf(int m, int n, dcomplex* a, int lda, int* ipiv);

        template<typename T>
        void inverse(const int& n, T* A);
}




class MatsubaraFreqGreen {
public:
	explicit MatsubaraFreqGreen(const size_t nw)
	: g_wn(nw,dcomplex(0,0)) 
	{
		_g_wn = &g_wn[0];
	}

	MatsubaraFreqGreen(const size_t nw, const dcomplex* g)
	: g_wn(g, g+nw) 
	{
		_g_wn = &g_wn[0];
	}

	MatsubaraFreqGreen(const MatsubaraFreqGreen& rhs)
	: g_wn(&rhs[0], &rhs[0]+rhs.get_nw())
	{
		_g_wn = &g_wn[0];
	}

	MatsubaraFreqGreen(const std::vector<dcomplex> rhs)
	: g_wn(&rhs[0], &rhs[0]+rhs.size())
	{
		_g_wn = &g_wn[0];
	}

	~MatsubaraFreqGreen() {}

	MatsubaraFreqGreen& operator=(const MatsubaraFreqGreen& rhs)
	{
		assert(g_wn.size() == rhs.size());
		std::memcpy(_g_wn,&rhs[0],sizeof(dcomplex)*g_wn.size());
		return *this;
	}

	MatsubaraFreqGreen operator*(const MatsubaraFreqGreen& rhs)
	{
		MatsubaraFreqGreen G(g_wn.size());
		for(size_t n=0;n<g_wn.size();++n) G[n] = g_wn[n] * rhs[n];
		return G;
	}

	int get_nw() const 
	{
		return g_wn.size();
	}

	int size() const 
	{
		return g_wn.size();
	}

	dcomplex& operator[](const size_t& i) const 
	{
		return _g_wn[i];
	}

	std::vector<dcomplex>::iterator begin()
	{
		return g_wn.begin();
	}

	std::vector<dcomplex>::iterator end()
	{
		return g_wn.end();
	}

	std::vector<dcomplex>::const_iterator begin() const
	{
		return g_wn.begin();
	}

	std::vector<dcomplex>::const_iterator end() const
	{
		return g_wn.end();
	}

	void set_iOmega_n(const double beta) 
	{
		for(size_t n=0;n<g_wn.size();++n) g_wn[n] = dcomplex(0,(2*n+1)*M_PI/beta);
	}

	MatsubaraFreqGreen operator-()
	{
		MatsubaraFreqGreen G(*this);
		for(auto& g : G) g *= -1;
		return G;
	}

	void print(const std::string FILE_NAME, const double BETA=1.)
	{
		std::ofstream file(FILE_NAME.c_str());
		for(int i=0;i<g_wn.size();++i)
		{
			double wn = (2*i+1)*M_PI/BETA;
			file<<wn<<" "<<g_wn[i].real()<<" "<<g_wn[i].imag()<<std::endl;
		}
	}

private:
	std::vector<dcomplex> g_wn;
	dcomplex* _g_wn = nullptr;
};

inline MatsubaraFreqGreen operator+(const MatsubaraFreqGreen& lhs,const MatsubaraFreqGreen& rhs) 
{
	const size_t nw = lhs.get_nw();
	MatsubaraFreqGreen Gf(nw);
	for(size_t i=0;i<nw;i++) Gf[i] = lhs[i] + rhs[i];
	return Gf;
}

inline MatsubaraFreqGreen operator-(const MatsubaraFreqGreen& lhs,const MatsubaraFreqGreen& rhs) 
{
	const size_t nw = lhs.get_nw();
	MatsubaraFreqGreen Gf(nw);
	for(size_t i=0;i<nw;i++) Gf[i] = lhs[i] - rhs[i];
	return Gf;
}

template<class T>
inline MatsubaraFreqGreen operator*(const MatsubaraFreqGreen& lhs,const T rhs) 
{
	const dcomplex factor = static_cast<dcomplex>(rhs);
	const size_t nw = lhs.get_nw();
	MatsubaraFreqGreen Gf(nw);
	for(size_t i=0;i<nw;i++) Gf[i] = lhs[i]*factor;
	return Gf;
}

template<class T>
inline MatsubaraFreqGreen operator*(const T lhs,const MatsubaraFreqGreen& rhs) 
{
	const dcomplex factor = static_cast<dcomplex>(lhs);
	const size_t nw = rhs.get_nw();
	MatsubaraFreqGreen Gf(nw);
	for(size_t i=0;i<nw;i++) Gf[i] = factor*rhs[i];
	return Gf;
}

template<class T>
inline MatsubaraFreqGreen operator+(const T lhs,const MatsubaraFreqGreen& rhs) 
{
	const dcomplex add = static_cast<dcomplex>(lhs);
	const size_t nw = rhs.get_nw();
	MatsubaraFreqGreen Gf(nw);
	for(size_t i=0;i<nw;i++) Gf[i] = add + rhs[i];
	return Gf;
}

template<class T>
inline MatsubaraFreqGreen operator+(const MatsubaraFreqGreen& lhs,const T rhs) 
{
	const dcomplex add = static_cast<dcomplex>(rhs);
	const size_t nw = lhs.get_nw();
	MatsubaraFreqGreen Gf(nw);
	for(size_t i=0;i<nw;i++) Gf[i] = lhs[i] + add;
	return Gf;
}

template<class T>
inline MatsubaraFreqGreen operator-(const T lhs,const MatsubaraFreqGreen& rhs) 
{
	const dcomplex sub = static_cast<dcomplex>(lhs);
	const size_t nw = rhs.get_nw();
	MatsubaraFreqGreen Gf(nw);
	for(size_t i=0;i<nw;i++) Gf[i] = sub - rhs[i];
	return Gf;
}

template<class T>
inline MatsubaraFreqGreen operator-(const MatsubaraFreqGreen& lhs,const T rhs) 
{
	const dcomplex sub = static_cast<dcomplex>(rhs);
	const size_t nw = lhs.get_nw();
	MatsubaraFreqGreen Gf(nw);
	for(size_t i=0;i<nw;i++) Gf[i] = lhs[i] - sub;
	return Gf;
}

inline MatsubaraFreqGreen inverse(const MatsubaraFreqGreen& Gin)
{
	const size_t nw = Gin.get_nw();
	MatsubaraFreqGreen Gf(nw);
	for(size_t i=0;i<nw;i++) Gf[i] = 1./Gin[i];
	return Gf;
}

inline MatsubaraFreqGreen hybridization(const int nw,const double beta,const double mu,const std::vector<double>& pvec) 
{
	MatsubaraFreqGreen Gf(nw);
	dcomplex iwn;
	const size_t nb = pvec.size()/2;
/*
   pvec[0]  ~ pvec[nb-1]  : eps_i
   pvec[nb] ~ pvec[2nb-1] : V_i
*/
	for(int n=0;n<nw;++n) 
	{
		iwn = dcomplex(0,(2*n+1)*M_PI/beta);
		Gf[n] = 0;
		for(int i=0;i<nb;i++) Gf[n] += pow(pvec[i+nb],2)/(iwn - pvec[i] + mu);
	}
	return Gf;
}

inline const std::ostream& operator<<(const std::ostream& lhs,const MatsubaraFreqGreen& Gin) 
{
	const size_t nw = Gin.get_nw();
	for(size_t n=0;n<5;n++) std::cout<<"G["<<n<<"] : "<<Gin[n]<<std::endl;
	std::cout<<"..."<<std::endl;
	std::cout<<"..."<<std::endl;
	std::cout<<"..."<<std::endl;
	std::cout<<"..."<<std::endl;
	for(size_t n=nw-5;n<nw;n++) std::cout<<"G["<<n<<"] : "<<Gin[n]<<std::endl;
	return lhs;
}

//Half-bandwidth := 1
inline MatsubaraFreqGreen create_BetheGf(const double beta,const double mu,const int nw) {
	MatsubaraFreqGreen BetheGf(nw);
	dcomplex iwn;
	for(size_t n=0;n<nw;++n) 
	{
		iwn = dcomplex(0,(2*n+1)*M_PI/beta);
		BetheGf[n] = dcomplex(2,0)*(iwn + mu - sqrt(pow(iwn+mu,2)-1.));
	}
	return BetheGf;
}


class MatsubaraTimeGreen
{
public:
	explicit MatsubaraTimeGreen(const size_t mesh)
	: g_tau(mesh,0)
	{
		_g_tau = g_tau.data();
	}

	MatsubaraTimeGreen(const size_t mesh, const double* rhs)
	: g_tau(rhs,rhs + mesh)
	{
		_g_tau = g_tau.data();
	}

	MatsubaraTimeGreen(const std::vector<double>& rhs)
	: g_tau(&rhs[0],&rhs[0] + rhs.size())
	{
		_g_tau = g_tau.data();
	}

	MatsubaraTimeGreen(const MatsubaraTimeGreen& rhs) 
	: g_tau(&rhs[0],&rhs[0] + rhs.size())
	{
		assert(g_tau.size() == rhs.size());
		_g_tau = g_tau.data();
		//for(size_t i=0;i<g_tau.size();++i) g_tau[i] = rhs[i];
	}

	MatsubaraTimeGreen& operator=(const MatsubaraTimeGreen& rhs)
	{
		//for(size_t i=0;i<g_tau.size();++i) g_tau[i] = rhs[i];
		assert(g_tau.size() == rhs.size());
		std::memcpy(&g_tau[0],&rhs[0],sizeof(double)*g_tau.size());
		return *this;
	}

	MatsubaraTimeGreen operator*(const MatsubaraTimeGreen& rhs)
	{
		MatsubaraTimeGreen G(g_tau.size());
		for(size_t n=0;n<g_tau.size();++n) G[n] = g_tau[n] * rhs[n];
		return G;
	}

	int size() const
	{
		return g_tau.size();
	}

	double& operator[](const size_t& i) const
	{
		return _g_tau[i];
	}

	std::vector<double>::iterator begin()
	{
		return g_tau.begin();
	}

	std::vector<double>::iterator end()
	{
		return g_tau.end();
	}

	std::vector<double>::const_iterator begin() const
	{
		return g_tau.begin();
	}

	std::vector<double>::const_iterator end() const
	{
		return g_tau.end();
	}

	MatsubaraTimeGreen operator-()
	{
		//MatsubaraTimeGreen G(g_tau.size());
		//for(size_t n=0;n<g_tau.size();++n) G[n] = -g_tau[n];
		MatsubaraTimeGreen G(*this);
		for(auto& g : G) g *= -1;
		return G;
	}

	void print(const std::string FILE_NAME, const double BETA=1.)
	{
		std::ofstream file(FILE_NAME.c_str());
		for(int i=0;i<g_tau.size();++i)
		{
			double tau = BETA*i/(g_tau.size() - 1.);
			file<<tau<<" "<<g_tau[i]<<std::endl;
		}
	}

private:
	std::vector<double> g_tau;
	double* _g_tau = nullptr;
};

inline MatsubaraTimeGreen operator+(const MatsubaraTimeGreen& lhs,const MatsubaraTimeGreen& rhs) 
{
	const size_t nw = lhs.size();
	MatsubaraTimeGreen Gf(nw);
	for(size_t i=0;i<nw;i++) Gf[i] = lhs[i] + rhs[i];
	return Gf;
}

inline MatsubaraTimeGreen operator-(const MatsubaraTimeGreen& lhs,const MatsubaraTimeGreen& rhs) 
{
	const size_t nw = lhs.size();
	MatsubaraTimeGreen Gf(nw);
	for(size_t i=0;i<nw;i++) Gf[i] = lhs[i] - rhs[i];
	return Gf;
}

template<class T>
inline MatsubaraTimeGreen operator*(const MatsubaraTimeGreen& lhs,const T rhs) 
{
	const double factor = static_cast<double>(rhs);
	const size_t nw = lhs.size();
	MatsubaraTimeGreen Gf(nw);
	for(size_t i=0;i<nw;i++) Gf[i] = lhs[i]*factor;
	return Gf;
}

template<class T>
inline MatsubaraTimeGreen operator*(const T lhs,const MatsubaraTimeGreen& rhs) 
{
	const double factor = static_cast<double>(lhs);
	const size_t nw = rhs.size();
	MatsubaraTimeGreen Gf(nw);
	for(size_t i=0;i<nw;i++) Gf[i] = factor*rhs[i];
	return Gf;
}

template<class T>
inline MatsubaraTimeGreen operator+(const T lhs,const MatsubaraTimeGreen& rhs) 
{
	const dcomplex add = static_cast<dcomplex>(lhs);
	const size_t nw = rhs.size();
	MatsubaraTimeGreen Gf(nw);
	for(size_t i=0;i<nw;i++) Gf[i] = add + rhs[i];
	return Gf;
}

template<class T>
inline MatsubaraTimeGreen operator+(const MatsubaraTimeGreen& lhs,const T rhs) 
{
	const dcomplex add = static_cast<dcomplex>(rhs);
	const size_t nw = lhs.size();
	MatsubaraTimeGreen Gf(nw);
	for(size_t i=0;i<nw;i++) Gf[i] = lhs[i] + add;
	return Gf;
}

template<class T>
inline MatsubaraTimeGreen operator-(const T lhs,const MatsubaraTimeGreen& rhs) 
{
	const dcomplex sub = static_cast<dcomplex>(lhs);
	const size_t nw = rhs.size();
	MatsubaraTimeGreen Gf(nw);
	for(size_t i=0;i<nw;i++) Gf[i] = sub - rhs[i];
	return Gf;
}

template<class T>
inline MatsubaraTimeGreen operator-(const MatsubaraTimeGreen& lhs,const T rhs) 
{
	const dcomplex sub = static_cast<dcomplex>(rhs);
	const size_t nw = lhs.size();
	MatsubaraTimeGreen Gf(nw);
	for(size_t i=0;i<nw;i++) Gf[i] = lhs[i] - sub;
	return Gf;
}

inline MatsubaraTimeGreen inverse(const MatsubaraTimeGreen& Gin)
{
	const size_t nw = Gin.size();
	MatsubaraTimeGreen Gf(nw);
	for(size_t i=0;i<nw;i++) Gf[i] = 1./Gin[i];
	return Gf;
}

inline const std::ostream& operator<<(const std::ostream& lhs,const MatsubaraTimeGreen& Gin) 
{
	const size_t nt = Gin.size();
	for(size_t n=0;n<5;n++) std::cout<<"G["<<n<<"] : "<<Gin[n]<<std::endl;
	std::cout<<"..."<<std::endl;
	std::cout<<"..."<<std::endl;
	std::cout<<"..."<<std::endl;
	std::cout<<"..."<<std::endl;
	for(size_t n=nt-5;n<nt;n++) std::cout<<"G["<<n<<"] : "<<Gin[n]<<std::endl;
	return lhs;
}



template<class TypeGreenFunc>
class MultiGreen
{
public:
	explicit MultiGreen(const std::initializer_list<TypeGreenFunc>& il)
	: G_wn(il) , flavor_name(il.size(),"None")
	{
		_G_wn = &G_wn[0];
	}

	explicit MultiGreen(const size_t dim,const size_t len)
	: G_wn(dim,TypeGreenFunc(len)) , flavor_name(dim,"None")
	{
		_G_wn = &G_wn[0];
	}

	MultiGreen(const MultiGreen& rhs)
	: G_wn(rhs.get_dim(),TypeGreenFunc(rhs[0].size()))
	{
		_G_wn = &G_wn[0];
		for(size_t i=0;i<G_wn.size();++i)
		{
			for(size_t n=0;n<(G_wn[0]).size();++n)
				_G_wn[i][n] = rhs[i][n];
		}
	}

	MultiGreen<TypeGreenFunc>& operator=(const MultiGreen<TypeGreenFunc>& rhs)
	{
		const size_t len = rhs.get_dim();
		const size_t nw = rhs[0].get_nw();
		for(size_t i=0;i<len;++i)
			for(size_t n=0;n<nw;++n)
				(*this)[i][n] = rhs[i][n];
		return *this;
	}

	MultiGreen<TypeGreenFunc> operator-()
	{
		MultiGreen<TypeGreenFunc> G(*this);
		for(auto& g : G)
			for(auto& g_wn : g)
				g_wn *= -1;
		return G;
	}

	TypeGreenFunc& operator[](const size_t i) const
	{
		return _G_wn[i];
	}

	size_t get_dim() const 
	{
		return G_wn.size();
	}

	typename std::vector<TypeGreenFunc>::iterator begin() 
	{
		return G_wn.begin();
	}

	typename std::vector<TypeGreenFunc>::iterator end() 
	{
		return G_wn.end();
	}

	typename std::vector<TypeGreenFunc>::const_iterator begin() const
	{
		return G_wn.begin();
	}

	typename std::vector<TypeGreenFunc>::const_iterator end() const
	{
		return G_wn.end();
	}

	void set_flavor_name(const std::initializer_list<std::string>& il)
	{
		try
		{
			if(il.size() != G_wn.size())
				throw std::runtime_error("check size of flavor name.");
		}	
		catch(const std::exception& err)
		{
			std::cout<<"Error:"<<err.what()<<std::endl;
		}
	}

	std::vector<std::string> flavor_name;
private:
	std::vector<TypeGreenFunc> G_wn;
	TypeGreenFunc* _G_wn = nullptr;
};

template<class TypeGreenFunc>
inline MultiGreen<TypeGreenFunc> operator+(const MultiGreen<TypeGreenFunc>& lhs,MultiGreen<TypeGreenFunc>& rhs)
{
	const size_t len = lhs.get_dim();
	const size_t nw = lhs[0].size();
	MultiGreen<TypeGreenFunc> G(len,nw);
	for(size_t i=0;i<len;++i)
		for(size_t n=0;n<nw;++n)
			G[i][n] = lhs[i][n] + rhs[i][n];
	return G;
}

template<class TypeGreenFunc,class TypeVector>
inline MultiGreen<TypeGreenFunc> operator+(const MultiGreen<TypeGreenFunc>& lhs,const std::vector<TypeVector>& rhs)
{
	const size_t len = lhs.get_dim();
	const size_t nw = lhs[0].size();
	MultiGreen<TypeGreenFunc> G(len,nw);

	for(size_t i=0;i<len;++i)
		for(size_t n=0;n<nw;++n)
			G[i][n] = lhs[i][n] + rhs[i];

	return G;
}

template<class TypeGreenFunc,class TypeVector>
inline MultiGreen<TypeGreenFunc> operator+(const std::vector<TypeVector>& lhs,const MultiGreen<TypeGreenFunc>& rhs)
{
	const size_t len = rhs.get_dim();
	const size_t nw = rhs[0].size();
	MultiGreen<TypeGreenFunc> G(len,nw);
	for(size_t i=0;i<len;++i)
		for(size_t n=0;n<nw;++n)
			G[i][n] = lhs[i] + rhs[i][n];
	return G;
}

template<class TypeGreenFunc>
inline MultiGreen<TypeGreenFunc> operator-(const MultiGreen<TypeGreenFunc>& lhs,MultiGreen<TypeGreenFunc>& rhs)
{
	const size_t len = lhs.get_dim();
	const size_t nw = lhs[0].size();
	MultiGreen<TypeGreenFunc> G(len,nw);
	for(size_t i=0;i<len;++i)
		for(size_t n=0;n<nw;++n)
			G[i][n] = lhs[i][n] - rhs[i][n];
	return G;
}

template<class TypeGreenFunc,class TypeVector>
inline MultiGreen<TypeGreenFunc> operator-(const MultiGreen<TypeGreenFunc>& lhs,const std::vector<TypeVector>& rhs)
{
	const size_t len = lhs.get_dim();
	const size_t nw = lhs[0].size();
	MultiGreen<TypeGreenFunc> G(len,nw);
	for(size_t i=0;i<len;++i)
		for(size_t n=0;n<nw;++n)
			G[i][n] = lhs[i][n] - rhs[i];
	return G;
}

template<class TypeGreenFunc,class TypeVector>
inline MultiGreen<TypeGreenFunc> operator-(const std::vector<TypeVector>& lhs,const MultiGreen<TypeGreenFunc>& rhs)
{
	const size_t len = rhs.get_dim();
	const size_t nw = rhs[0].size();
	MultiGreen<TypeGreenFunc> G(len,nw);
	for(size_t i=0;i<len;++i)
		for(size_t n=0;n<nw;++n)
			G[i][n] = lhs[i] - rhs[i][n];
	return G;
}

template<class TypeGreenFunc>
inline MultiGreen<TypeGreenFunc> operator*(const MultiGreen<TypeGreenFunc>& lhs,const MultiGreen<TypeGreenFunc>& rhs)
{
	const size_t dim = lhs.get_dim();
	const size_t nw = lhs[0].size();
	const size_t row = std::sqrt(dim);
	std::vector<std::vector<dcomplex> > comp_lhs(nw,std::vector<dcomplex>(dim,dcomplex(0,0)));
	std::vector<std::vector<dcomplex> > comp_rhs(nw,std::vector<dcomplex>(dim,dcomplex(0,0)));
	std::vector<std::vector<dcomplex> > comp(nw,std::vector<dcomplex>(dim,dcomplex(0,0)));

	for(size_t n=0;n<nw;++n)
		for(size_t i=0;i<dim;++i)
		{
			comp_lhs[n][i] = lhs[i][n];
			comp_rhs[n][i] = rhs[i][n];
		}

	for(size_t n=0;n<nw;++n)
		for(size_t i=0;i<row;++i)
			for(size_t j=0;j<row;j++)
				for(size_t k=0;k<row;k++)
				{
					comp[n][row*i+j] += comp_lhs[n][row*i+k] * comp_rhs[n][row*k+j];
				}

	MultiGreen<MatsubaraFreqGreen> G(dim,nw);

	for(size_t n=0;n<nw;++n)
		for(size_t i=0;i<dim;++i)
			G[i][n] = comp[n][i];
	return G;
}

template<class TypeGreenFunc,class TypeVector>
inline MultiGreen<TypeGreenFunc> operator*(const MultiGreen<TypeGreenFunc>& lhs,const std::vector<TypeVector>& rhs)
{
	const size_t dim = lhs.get_dim();
	const size_t nw = lhs[0].size();
	const size_t row = std::sqrt(dim);
	std::vector<std::vector<dcomplex> > comp_lhs(nw,std::vector<dcomplex>(dim,dcomplex(0,0)));
	std::vector<std::vector<dcomplex> > comp(nw,std::vector<dcomplex>(dim,dcomplex(0,0)));

	for(size_t n=0;n<nw;++n)
		for(size_t i=0;i<dim;++i)
			comp_lhs[n][i] = lhs[i][n];

	for(size_t n=0;n<nw;++n)
		for(size_t i=0;i<row;++i)
			for(size_t j=0;j<row;j++)
				for(size_t k=0;k<row;k++)
				{
					comp[n][row*i+j] += comp_lhs[n][row*i+k] * rhs[row*k+j];
				}

	MultiGreen<MatsubaraFreqGreen> G(dim,nw);

	for(size_t n=0;n<nw;++n)
		for(size_t i=0;i<dim;++i)
			G[i][n] = comp[n][i];
	return G;
}

template<class TypeGreenFunc,class TypeVector>
inline MultiGreen<TypeGreenFunc> operator*(const std::vector<TypeVector>& lhs,const MultiGreen<TypeGreenFunc>& rhs)
{
	const size_t dim = rhs.get_dim();
	const size_t nw = rhs[0].size();
	const size_t row = std::sqrt(dim);
	std::vector<std::vector<dcomplex> > comp_rhs(nw,std::vector<dcomplex>(dim,dcomplex(0,0)));
	std::vector<std::vector<dcomplex> > comp(nw,std::vector<dcomplex>(dim,dcomplex(0,0)));

	for(size_t n=0;n<nw;++n)
		for(size_t i=0;i<dim;++i)
			comp_rhs[n][i] = rhs[i][n];

	for(size_t n=0;n<nw;++n)
		for(size_t i=0;i<row;++i)
			for(size_t j=0;j<row;j++)
				for(size_t k=0;k<row;k++)
				{
					comp[n][row*i+j] += lhs[row*i+k] *comp_rhs[n][row*k+j] ;
				}

	MultiGreen<MatsubaraFreqGreen> G(dim,nw);

	for(size_t n=0;n<nw;++n)
		for(size_t i=0;i<dim;++i)
			G[i][n] = comp[n][i];
	return G;
}

template<class TypeGreenFunc,class TypeScalar>
inline MultiGreen<TypeGreenFunc> operator*(const MultiGreen<TypeGreenFunc>& lhs,const TypeScalar& rhs)
{
	const size_t len = lhs.get_dim();
	const size_t nw = lhs[0].size();
	MultiGreen<TypeGreenFunc> G(len,nw);
	for(size_t i=0;i<len;++i)
		for(size_t n=0;n<nw;++n)
			G[i][n] = lhs[i][n] * rhs;
	return G;
}

template<class TypeGreenFunc,class TypeScalar>
inline MultiGreen<TypeGreenFunc> operator*(const TypeScalar& lhs,const MultiGreen<TypeGreenFunc>& rhs)
{
	const size_t len = rhs.get_dim();
	const size_t nw = rhs[0].size();
	MultiGreen<TypeGreenFunc> G(len,nw);
	for(size_t i=0;i<len;++i)
		for(size_t n=0;n<nw;++n)
			G[i][n] = lhs * rhs[i][n];
	return G;
}

// dimension of MultiGreen should be a square to row or column length of matrix.
inline MultiGreen<MatsubaraFreqGreen> inverse(const MultiGreen<MatsubaraFreqGreen>& rhs)
{
	const size_t dim = rhs.get_dim();
	const size_t nw = rhs[0].size();
	std::vector<std::vector<dcomplex> > component(nw,std::vector<dcomplex>(dim,dcomplex(0,0)));

	for(size_t n=0;n<nw;++n)
	{
		for(size_t i=0;i<dim;++i)
			component[n][i] = rhs[i][n];
		lapack::inverse(std::sqrt(dim),component[n].data());
	}

	MultiGreen<MatsubaraFreqGreen> G(dim,nw);

	for(size_t n=0;n<nw;++n)
		for(size_t i=0;i<dim;++i)
			G[i][n] = component[n][i];
	return G;
}

template <class TypeGreenFunc>
inline const std::ostream& operator<<(const std::ostream& lhs,const MultiGreen<TypeGreenFunc>& G) 
{
	const size_t dim = G.get_dim();
	for(size_t i=0;i<dim;++i)
	{
		std::cout<<"#"<<G.flavor_name[i]<<std::endl;
		std::cout<<G[i];
	}
	return lhs;
}


// FFTW3 library.
// FFTW_DIRECTION : FORWARD(-1) , BACKWARD(+1)

enum FOURIER_TRANSFORM { FORWARD = -1, BACKWARD = 1 };

// Fast Fourier transform for 1-D.
void fft_1d_complex(const dcomplex* input, dcomplex* output, const int N, const FOURIER_TRANSFORM FFTW_DIRECTION)
{
	fftw_complex *in, *out;
	fftw_plan p;

	in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N); 
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N); 
	p = fftw_plan_dft_1d(N, in, out, FFTW_DIRECTION, FFTW_ESTIMATE);

	for(int i=0;i<N;++i)
	{
		in[i][0] = input[i].real();
		in[i][1] = input[i].imag();
	}   

	fftw_execute(p); // repeat as needed 

	for(int i=0;i<N;++i)
		output[i] = dcomplex(out[i][0],out[i][1]);

	fftw_destroy_plan(p);

	fftw_free(in); fftw_free(out);
}


MatsubaraTimeGreen InverseFourier(const MatsubaraFreqGreen& G_wn,const double beta,const std::vector<double> M, const size_t mesh=1e4)
{
	const size_t Niwn = G_wn.size();
	double dtau = beta/(mesh-1);
	double tau = 0;
	MatsubaraTimeGreen G_tau(mesh);
	dcomplex iwn,g_tau;
	for(size_t i=0;i<mesh;++i)	
	{
		g_tau = dcomplex(0,0);
		tau = dtau*i;
		for(size_t n=0;n<Niwn;++n)
		{
			iwn = dcomplex(0,(2*n+1)*M_PI/beta);
			g_tau += std::exp(-iwn*tau)*(G_wn[n] - (M[0]/iwn + M[1]/std::pow(iwn,2) + M[2]/std::pow(iwn,3)));
		}
		g_tau *= 2./beta;
		g_tau += -0.5*M[0] + (tau/2.-beta/4.)*M[1] + (tau*beta/4. - std::pow(tau,2)/4.)*M[2];
		G_tau[i] = g_tau.real();
	}
	return G_tau;
}


// Inverse Fourier transform for the fermionic Matsubara green function.
std::vector<double> fft_InverseFourier(const dcomplex* G_wn, const size_t WN_MESH, const double BETA, const std::vector<double> M, const size_t TAU_MESH)
{
	// check that mesh size satisfies the Nyquist theorem or not.
	assert(TAU_MESH/2 >= WN_MESH);

	auto dist = [&WN_MESH,&BETA,&M](const dcomplex g_fin)->double
		{
			dcomplex iwn(0,(2.*WN_MESH-1.)*M_PI/BETA);
			return std::norm(g_fin - (M[0]/iwn + M[1]/std::pow(iwn,2) + M[2]/std::pow(iwn,3)));
		};

	if(dist(G_wn[WN_MESH-1]) >= 1e-7)
		std::cout<<"warning! enlarge your size of the Matsubara frequency region!(fourier transform can give a result with a large error...)"<<std::endl;

	std::vector<dcomplex> iwn(WN_MESH);
	for(int i=0;i<WN_MESH;++i) iwn[i] = dcomplex(0,(2*i+1)*M_PI/BETA);

	std::vector<dcomplex> g_wn(TAU_MESH,dcomplex(0,0));
	std::vector<dcomplex> g_temp(TAU_MESH);

	for(int i=0;i<WN_MESH;++i)
		g_wn[i] = (G_wn[i] - (M[0]/iwn[i] + M[1]/std::pow(iwn[i],2) + M[2]/std::pow(iwn[i],3)));
		// g_wn for i >= WN_MESH is 0.

	fft_1d_complex(g_wn.data(),g_temp.data(),TAU_MESH-1,FORWARD);

	
	for(int i=0;i<TAU_MESH-1;++i)
		g_temp[i] += g_wn[TAU_MESH-1];
	

	std::vector<double> g_tau(TAU_MESH,0);

	for(int i=0;i<TAU_MESH-1;++i)
	{
		double tau = BETA*i/(TAU_MESH-1.);
		g_tau[i] = 2./BETA*(std::exp(dcomplex(0,-M_PI*tau/BETA))*g_temp[i]).real() -
			        0.5*M[0] + (tau/2.-BETA/4.)*M[1] + (tau*BETA/4. - std::pow(tau,2)/4.)*M[2];
	}

	auto g_edge = dcomplex(0,0); // := G(BETA)

	for(int i=0;i<WN_MESH;++i)
		g_edge += -1.*g_wn[i];
		// g_wn for i >= WN_MESH is 0.
		
	g_edge *= 2./BETA;

	g_tau[TAU_MESH-1] = g_edge.real() - 0.5*M[0] + (BETA/4)*M[1];

	return g_tau;
}


// Fourier transform for the fermionic Matsubara green function.
std::vector<dcomplex> fft_Fourier(const double* G_tau, const size_t TAU_MESH, const double BETA, const size_t WN_MESH)
{
/*std::vector<dcomplex> fft_Fourier(const double* G_tau, const size_t TAU_MESH, const double BETA, const std::vector<double> M, const size_t WN_MESH)
{
*/
	// check that mesh size satisfies the Nyquist theorem or not.
	assert(TAU_MESH/2 >= WN_MESH);
	const double dTau = BETA/(TAU_MESH - 1.);

	// The Simpson integration law is applied. ----> In our implementation, something goes wrong. It should be fixed... :( 
	auto generSimpsWeight = [](dcomplex* gt, const double dTau, const size_t TAU_MESH) -> void
			{
				/*
				assert(TAU_MESH%2 ==0);
                		for(int i=1;i<TAU_MESH; i+=2) gt[i] *= 4.;
                		for(int i=2;i<TAU_MESH-1; i+=2) gt[i] *= 2;
				for(int i=0;i<TAU_MESH;++i)
					gt[i] *= dTau/3.;
				*/ 
				for(int i=0;i<TAU_MESH;++i)
					gt[i] *= dTau;
			};

	std::vector<dcomplex> g_mod(TAU_MESH,0);

	for(int i=0;i<TAU_MESH;++i)
	{
		double tau = BETA*i/(TAU_MESH - 1.);
		//g_mod[i] = G_tau[i] - (-M[0]/2. + M[1]*(tau/2. - BETA/4.) + M[2]*(tau*BETA/4. - std::pow(tau,2)/4.));
		g_mod[i] = G_tau[i];
		g_mod[i] *= std::exp(dcomplex(0,M_PI/BETA*tau));
	}

	generSimpsWeight(&g_mod[0],dTau,g_mod.size());

	std::vector<dcomplex> g_temp(TAU_MESH,dcomplex(0,0));

	fft_1d_complex(g_mod.data(),g_temp.data(),TAU_MESH-1,BACKWARD);
	
	for(int i=0;i<TAU_MESH-1;++i)
		g_temp[i] += g_mod[TAU_MESH-1];

	std::vector<dcomplex> iwn(WN_MESH);
	for(int i=0;i<WN_MESH;++i)
		iwn[i] = dcomplex(0,(2*i+1)*M_PI/BETA);


	std::vector<dcomplex> g_iwn(&g_temp[0],&g_temp[0] + WN_MESH);

	/*
	std::vector<dcomplex> g_iwn(WN_MESH);
	for(int i=0;i<WN_MESH;++i)
		g_iwn[i] = g_temp[i] + (M[0]/iwn[i] + M[1]/std::pow(iwn[i],2) + M[2]/std::pow(iwn[i],3));
	*/

	return g_iwn;
}

/*
MultiGreen<MatsubaraTimeGreen> InverseFourier(const MultiGreen<MatsubaraFreqGreen>& G_wn,const double beta,const size_t mesh=1e4)
{
	MultiGreen<MatsubaraTimeGreen> G_tau(G_wn.get_dim(),mesh);
	for(size_t i=0;i<G_wn.get_dim();++i) G_tau[i] = InverseFourier(G_wn[i]);
	return G_tau;
}
*/

double density(const MatsubaraFreqGreen& G_wn,const double beta,const std::vector<double>& M)
{
	const size_t Niwn = G_wn.size();
	dcomplex iwn;
	dcomplex dens = 0;

	for(size_t n=0;n<Niwn;++n)
	{
		iwn = dcomplex(0,(2*n+1)*M_PI/beta);
		dens += (G_wn[n] - (M[0]/iwn + M[1]/std::pow(iwn,2) + M[2]/std::pow(iwn,3)));
	}
	dens *= 2./beta;
	dens += -0.5*M[0] + (-beta/4.)*M[1];

	return 1.+dens.real();
}

/*
MultiGreen<MatsubaraFreqGreen> Fourier(const MultiGreen<MatsubaraTimeGreen>& G_tau,const size_t nw=1025)
{
	MultiGreen<MatsubaraFreqGreen> G_wn(G_tau.get_dim(),nw);
	for(size_t i=0;i<G_tau.get_dim();++i) G_wn[i] = Fourier(G_tau[i]);
	return G_wn;
}

MatsubaraFreqGreen Fourier(const MatsubaraTimeGreen& G_tau,const size_t nw=1025)
{
	MatsubaraFreqGreen G_wn(nw);
	// ...
	return G_wn;
}
*/


void lapack::getrf(int m, int n, float* a, int* ipiv)
{
        int info;
        sgetrf_(&m,&n,a,&n,ipiv,&info);
        assert(info==0);
}

void lapack::getrf(int m, int n, double* a, int* ipiv)
{
        int info;
        dgetrf_(&m,&n,a,&n,ipiv,&info);
        assert(info==0);
}

void lapack::getrf(int m, int n, scomplex* a, int* ipiv)
{
        int info;
        cgetrf_(&m,&n,a,&n,ipiv,&info);
        assert(info==0);
}

void lapack::getrf(int m, int n, dcomplex* a, int* ipiv)
{
        int info;
        zgetrf_(&m,&n,a,&n,ipiv,&info);
        assert(info==0);
}

void lapack::getrf(int m, int n, float* a, int lda, int* ipiv)
{
        int info;
        sgetrf_(&m,&n,a,&lda,ipiv,&info);
        assert(info==0);
}

void lapack::getrf(int m, int n, double* a, int lda, int* ipiv)
{
        int info;
        dgetrf_(&m,&n,a,&lda,ipiv,&info);
        assert(info==0);
}

void lapack::getrf(int m, int n, scomplex* a, int lda, int* ipiv)
{
        int info;
        cgetrf_(&m,&n,a,&lda,ipiv,&info);
        assert(info==0);
}

void lapack::getrf(int m, int n, dcomplex* a, int lda, int* ipiv)
{
        int info;
        zgetrf_(&m,&n,a,&lda,ipiv,&info);
        assert(info==0);
}

void lapack::getri(int n, float* a, int* ipiv)
{
        int info, lwork=-1;
        float tmp;
        sgetri_(&n,a,&n,ipiv,&tmp,&lwork,&info);
        lwork=(int)(tmp+0.1);
        float* work=new float[lwork];
        sgetri_(&n,a,&n,ipiv,work,&lwork,&info);
        delete[] work;
        assert(info==0);
}

void lapack::getri(int n, double* a, int* ipiv)
{
        int info, lwork=-1;
        double tmp;
        dgetri_(&n,a,&n,ipiv,&tmp,&lwork,&info);
        lwork=(int)(tmp+0.1);
        double* work=new double[lwork];
        dgetri_(&n,a,&n,ipiv,work,&lwork,&info);
        delete[] work;
        assert(info==0);
}

void lapack::getri(int n, scomplex* a, int* ipiv)
{
        int info, lwork=-1;
        scomplex tmp;
        cgetri_(&n,a,&n,ipiv,&tmp,&lwork,&info);
        lwork=(int)(tmp.real()+0.1);
        scomplex* work=new scomplex[lwork];
        cgetri_(&n,a,&n,ipiv,work,&lwork,&info);
        delete[] work;
        assert(info==0);
}

void lapack::getri(int n, dcomplex* a, int* ipiv)
{
        int info, lwork=-1;
        dcomplex tmp;
        zgetri_(&n,a,&n,ipiv,&tmp,&lwork,&info);
        lwork=(int)(tmp.real()+0.1);
        dcomplex* work=new dcomplex[lwork];
        zgetri_(&n,a,&n,ipiv,work,&lwork,&info);
        delete[] work;
        assert(info==0);
}

void lapack::getri(int n, float* a, int lda, int* ipiv)
{
        int info, lwork=-1;
        float tmp;
        sgetri_(&n,a,&lda,ipiv,&tmp,&lwork,&info);
        lwork=(int)(tmp+0.1);
        float* work=new float[lwork];
        sgetri_(&n,a,&lda,ipiv,work,&lwork,&info);
        delete[] work;
        assert(info==0);
}

void lapack::getri(int n, double* a, int lda, int* ipiv)
{
        int info, lwork=-1;
        double tmp;
        dgetri_(&n,a,&lda,ipiv,&tmp,&lwork,&info);
        lwork=(int)(tmp+0.1);
        double* work=new double[lwork];
        dgetri_(&n,a,&lda,ipiv,work,&lwork,&info);
        delete[] work;
        assert(info==0);
}

void lapack::getri(int n, scomplex* a, int lda, int* ipiv)
{
        int info, lwork=-1;
        scomplex tmp;
        cgetri_(&n,a,&lda,ipiv,&tmp,&lwork,&info);
        lwork=(int)(tmp.real()+0.1);
        scomplex* work=new scomplex[lwork];
        cgetri_(&n,a,&lda,ipiv,work,&lwork,&info);
        delete[] work;
        assert(info==0);
}

void lapack::getri(int n, dcomplex* a, int lda, int* ipiv)
{
        int info, lwork=-1;
        dcomplex tmp;
        zgetri_(&n,a,&lda,ipiv,&tmp,&lwork,&info);
        lwork=(int)(tmp.real()+0.1);
        dcomplex* work=new dcomplex[lwork];
        zgetri_(&n,a,&lda,ipiv,work,&lwork,&info);
        delete[] work;
        assert(info==0);
}

template<typename T>
void lapack::inverse(const int& n, T* A) { 
        if (n==1) {
                A[0]=1.0/A[0];
        }
        if (n==2) {
                T Ainv[4];
                const T fac=1.0/(A[0]*A[3]-A[1]*A[2]);
                Ainv[0]=A[3]*fac;
                Ainv[1]=-A[1]*fac;
                Ainv[2]=-A[2]*fac;
                Ainv[3]=A[0]*fac;
                for (int i=0;i<4;++i) A[i]=Ainv[i];
        }
        if (n==3) {
                T Ainv[9];
                const T fac=1.0/(A[0]*(A[8]*A[4]-A[5]*A[7])
                                -A[1]*(A[8]*A[3]-A[5]*A[6])
                                +A[2]*(A[7]*A[3]-A[4]*A[6]));
                Ainv[0]=(A[8]*A[4]-A[5]*A[7])*fac;
                Ainv[1]=-(A[8]*A[1]-A[2]*A[7])*fac;
                Ainv[2]=(A[5]*A[1]-A[2]*A[4])*fac;
                Ainv[3]=-(A[8]*A[3]-A[5]*A[6])*fac;
                Ainv[4]=(A[8]*A[0]-A[2]*A[6])*fac;
                Ainv[5]=-(A[5]*A[0]-A[2]*A[3])*fac;
                Ainv[6]=(A[7]*A[3]-A[4]*A[6])*fac;
                Ainv[7]=-(A[7]*A[0]-A[1]*A[6])*fac;
                Ainv[8]=(A[4]*A[0]-A[1]*A[3])*fac;
                for (int i=0;i<9;++i) A[i]=Ainv[i];
        }
        if (n>3) {
                int* ipiv=new int[n];
                lapack::getrf(n,n,A,ipiv);
                lapack::getri(n,A,ipiv);
                delete[] ipiv;
        }
}

#endif
