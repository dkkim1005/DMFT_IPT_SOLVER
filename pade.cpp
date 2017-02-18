#include "pade.h"

/*
	Ref: Žiga Osolin and Rok Žitko
	     Phys. Rev. B 87, 245135(2013)
*/


gmp_complex gmp_pow(const gmp_complex& a, const int NUM_POWER, const int& PRECISION)
{
	gmp_complex temp(GC(1,0,PRECISION));
	for(int i=0;i<NUM_POWER;++i)
		temp *= a;
	return temp;
}


void gmp_gemv(const int M, const int N, const gmp_complex& ALPHA, gmp_complex* a, gmp_complex* x, const gmp_complex& BETA, gmp_complex* y, const int PRECISION)
{
        for(int i=0;i<M;++i)
        {
                gmp_complex temp(mpf_class(0,PRECISION),mpf_class(0,PRECISION));
                for(int j=0;j<N;++j)
                {
                        temp += a[i*N+j]*x[j];
                }
                temp *= ALPHA;
                y[i] = temp + BETA*y[i];
        }    
}


void arraySwap(gmp_complex* a1, gmp_complex* a2, const int numLength, const int& PRECISION)
{
	gmp_complex temp(mpf_class(0,PRECISION),mpf_class(0,PRECISION));

	for(int i=0;i<numLength;++i)
	{
		temp = a1[i];   
		a1[i] = a2[i];
		a2[i] = temp;
	}
}


bool gauss_jordan(const std::vector<gmp_complex>& inMatrix, std::vector<gmp_complex>& outMatrix, const int LENGTH, const int PRECISION)
{
	const int LOW_POINT_DECIMAL = static_cast<int>(PRECISION*std::log(2)/std::log(10)) - 1;
	assert(LOW_POINT_DECIMAL > 15);

	std::string zero_str("0.");
	for(int i=0;i<LOW_POINT_DECIMAL-1;++i) zero_str += "0";
	zero_str += "1";

	const mpf_class ZERO(zero_str.c_str(),PRECISION);

	std::vector<gmp_complex> lhs(LENGTH*LENGTH,gmp_complex(mpf_class(0,PRECISION),mpf_class(0,PRECISION)));
	std::vector<gmp_complex>& rhs = outMatrix;

	lhs = inMatrix;

	for(int i=0;i<LENGTH;++i) rhs[i*LENGTH+i] = 1.;

	for(int i=0;i<LENGTH;++i)
	{
		if( cmp(std::abs(lhs[i*LENGTH+i]),ZERO) < 0 )
		{
			int pivot = -1;
			for(int j=i+1;j<LENGTH;++j)
			{
				if( cmp(std::abs(lhs[j*LENGTH+i]),ZERO) > 0 )
				{
					pivot = j;
					break;
				}
			}

			if(pivot < 0) goto SINGULAR;

			arraySwap(&lhs[i*LENGTH],&lhs[pivot*LENGTH],LENGTH,PRECISION);
			arraySwap(&rhs[i*LENGTH],&rhs[pivot*LENGTH],LENGTH,PRECISION);
		}

		const gmp_complex HEADER = lhs[i*LENGTH+i];

		for(int j=i;j<LENGTH;++j) lhs[i*LENGTH+j] /= HEADER;
		for(int j=0;j<LENGTH;++j) rhs[i*LENGTH+j] /= HEADER;

		for(int j=i+1;j<LENGTH;++j)
		{
			const gmp_complex FACTOR = lhs[j*LENGTH+i];
			for(int k=i;k<LENGTH;++k)
				lhs[j*LENGTH+k] -= FACTOR*lhs[i*LENGTH+k];
			for(int k=0;k<LENGTH;++k)
				rhs[j*LENGTH+k] -= FACTOR*rhs[i*LENGTH+k];
		}
	}

	for(int i=LENGTH-1;i>0;--i)
		for(int j=0;j<i;++j)
			for(int k=0;k<LENGTH;++k)
				rhs[j*LENGTH+k] -= lhs[j*LENGTH+i]*rhs[i*LENGTH+k];

	return true;


SINGULAR:

	std::cout<<"matrix is singular!"<<std::endl;
	return false;	
}


void PadeForGreenFunction::padeApproximation(const dcomplex* z_i, const dcomplex* f_i, const int LENGTH, const int PRECISION, gmp_complex* x)
{
	assert(LENGTH%2 == 0); // LENGTH = 2*R

	const int R = LENGTH/2;
	std::vector<gmp_complex> A(LENGTH*LENGTH,GC(0,0,PRECISION));

	for(int i=0;i<LENGTH;++i)
	{
		for(int j=0;j<R;++j)
			A[i*LENGTH+j] = gmp_pow(GC(z_i[i].real(),z_i[i].imag(),PRECISION),j,PRECISION);

		for(int j=R;j<LENGTH;++j)
			A[i*LENGTH+j] = -GC(f_i[i].real(),f_i[i].imag(),PRECISION)*gmp_pow(GC(z_i[i].real(),z_i[i].imag(),PRECISION),j-R,PRECISION);
	}

	std::vector<gmp_complex> inv_A(LENGTH*LENGTH,GC(0,0,PRECISION));

	bool is_Inv_A_exists = gauss_jordan(A,inv_A,LENGTH,PRECISION);

	assert(is_Inv_A_exists);

	std::vector<gmp_complex> b(LENGTH,GC(0,0,PRECISION));
	for(int i=0;i<LENGTH;++i)
		b[i] = GC(f_i[i].real(),f_i[i].imag(),PRECISION)*gmp_pow(GC(z_i[i].real(),z_i[i].imag(),PRECISION),R,PRECISION);

	gmp_gemv(LENGTH,LENGTH,GC(1,0,PRECISION),&inv_A[0],&b[0],GC(0,0,PRECISION),&x[0],PRECISION);
}


gmp_complex PadeForGreenFunction::padeModelling(const gmp_complex& z, const gmp_complex* x, const int LENGTH, const int PRECISION) const
{
	assert(LENGTH%2 == 0); // LENGTH = 2*R

	const int R = LENGTH/2;
	gmp_complex numerator(mpf_class(0,PRECISION),mpf_class(0,PRECISION));
	gmp_complex denominator(mpf_class(0,PRECISION),mpf_class(0,PRECISION));

	for(int i=0;i<R;++i)
	{
		numerator += x[i]*gmp_pow(z,i,PRECISION);
		denominator += x[i+R]*gmp_pow(z,i,PRECISION);
	}

	denominator += gmp_pow(z,R,PRECISION);

	return numerator/denominator;
}

