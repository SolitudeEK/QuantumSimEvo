#include "Dirac.h"
#include <complex>
#include <vector>

using Complex = std::complex<double>;

std::vector<Complex> Dirac::ket(size_t dim, size_t a) {
	if (a >= dim) throw std::out_of_range("Index a exceeds dimension.");

	std::vector<Complex> k(dim);
	k[a] = { 1.0};
	return k;
}

std::vector<Complex> Dirac::bra(size_t dim, size_t a) {
	return  ket(a, dim);
}

Complex Dirac::braKet(size_t dim, size_t a, size_t b) {
	return (a == b) ? Complex{ 1.0} : Complex{ 0.0 };
}

void Dirac::ketBra(std::vector<Complex>& state, size_t a, size_t b) {
	Complex coef = state[b];

	std::fill(state.begin(), state.end(), 0.0);
	state[a] = coef;
}
