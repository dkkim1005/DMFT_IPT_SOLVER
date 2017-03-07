/*
	Ref: Žiga Osolin and Rok Žitko
	     Phys. Rev. B 87, 245135(2013)
*/

#if !defined(__PADE_ANALYTIC_CONTINUATION__)
#define __PADE_ANALYTIC_CONTINUATION__

//#define MAKE_PADE_PYTHON
#if defined(MAKE_PADE_PYTHON)
#include <boost/python.hpp>
#include "numpy_array.h"
#endif

#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <assert.h>
#include <gmpxx.h>
#include <fstream>
#include <cstring>

typedef std::complex<double> dcomplex;
typedef std::complex<mpf_class> gmp_complex;

#define GC(a,b,PREC) gmp_complex(mpf_class(a,PREC),mpf_class(b,PREC))


class PadeForGreenFunction
{
public:

	PadeForGreenFunction(const int PRECISION)
	: _precision(PRECISION), _is_assigned_x(false) {};

	void Fitting(const dcomplex* z, const dcomplex* g, const int LENGTH)
	{
		_length = LENGTH;
		_x.resize(_length,GC(0,0,_precision));

		padeApproximation(&z[0],&g[0],_length,_precision,&_x[0]);
		_is_assigned_x = true;
	}

	dcomplex estimate(const dcomplex& Z) const
	{
		assert(_is_assigned_x == true);
		gmp_complex estimated_g = padeModelling(GC(Z.real(),Z.imag(),_precision),&_x[0],_length,_precision);
		dcomplex result = dcomplex((estimated_g.real()).get_d(),(estimated_g.imag()).get_d());
		return result;
	}

	void setPrecision(const int PRECISION) 
	{
		_precision = PRECISION;
		std::vector<gmp_complex>().swap(_x);
		_is_assigned_x = false;
	}

private:

	void padeApproximation(const dcomplex* z_i, const dcomplex* f_i, const int LENGTH, const int PRECISION, gmp_complex* x);

	gmp_complex padeModelling(const gmp_complex& z, const gmp_complex* x, const int LENGTH, const int PRECISION) const;


	bool _is_assigned_x;
	std::vector<gmp_complex> _x;
	int _length;
	int _precision;
};


#if defined(MAKE_PADE_PYTHON)

class PythonWrapperForPade
{
public:
	PythonWrapperForPade(int precision)
	: Pade(precision) {}

	void Fitting(PyObject* z, PyObject* g)
	{
		ndarray_to_C_ptr_wrapper<dcomplex> cpp_z(z);
		ndarray_to_C_ptr_wrapper<dcomplex> cpp_g(g);

		const int LENGTH = cpp_z.get_total_size();
		Pade.Fitting(&cpp_z[0],&cpp_g[0],LENGTH);
	}

	PyObject* estimate(PyObject* z)
	{
		ndarray_to_C_ptr_wrapper<dcomplex> cpp_z(z);
		const int LENGTH = cpp_z.get_total_size();

		std::vector<dcomplex> result(LENGTH,dcomplex(0,0));

		for(int i=0;i<LENGTH;++i)
			result[i] = Pade.estimate(cpp_z[i]);

		npy_intp dims[1] = {LENGTH};

		return C_ptr_to_ndarray_wrapper(&result[0],1,&dims[0],NPY_COMPLEX128);
	}

	void setPrecision(const int PRECISION) { Pade.setPrecision(PRECISION); }

private:
	PadeForGreenFunction Pade;
};


BOOST_PYTHON_MODULE(AC_Pade)
{
	import_array();
	boost::python::class_<PythonWrapperForPade>("Pade_method",boost::python::init<int>())
		.def("Fitting",&PythonWrapperForPade::Fitting)
		.def("estimate",&PythonWrapperForPade::estimate)
		.def("setPrecision",&PythonWrapperForPade::setPrecision)
	;
}

#endif


#endif
