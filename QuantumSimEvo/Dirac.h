#ifndef DIRAC_H
#define DIRAC_H
#include <vector>
#include <complex>

using Complex = std::complex<double>;

class Dirac {
	public:
	static std::vector<Complex> ket(size_t dim, size_t a);
	static std::vector<Complex> bra(size_t dim, size_t a);
	static Complex braKet(size_t dim, size_t a, size_t b);
	static void ketBra(std::vector<Complex>& state, size_t a, size_t b);
};

#endif