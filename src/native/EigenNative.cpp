// EigenNative.cpp : Defines the entry point for the application.

#include "EigenNative.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

using namespace std;
using namespace Eigen;

// dot product between two vectors.
EXPORT_API(double) ddot_(_In_  double* v1, _In_  double* v2, int length1)
{
	Map<const VectorXd> first(v1, length1);
	Map<const VectorXd> second(v2, length1);
	return first.dot(second);
}

// addition of two vectors.
EXPORT_API(void) dadd_(_In_ double* v1, _In_ double* v2, int length1, _Out_ double* vout)
{
	Map<const VectorXd> first(v1, length1);
	Map<const VectorXd> second(v2, length1);
	Map<VectorXd>  result(vout, length1);
	result = first + second;
}


// scale a vector by a scalar.
EXPORT_API(void) dscale_(_In_ double* v1, double scale, int length1, _Out_ double* vout)
{
	Map<const VectorXd> first(v1, length1);
	Map<VectorXd> result(vout, length1);
	result = first * scale;
}

// m1 - m2.
EXPORT_API(void) dminus_(_In_ double* m1, const int row1, const int col1, _In_ double* m2, const int row2, const int col2, _Out_ double* vout)
{
	Map<const MatrixXd> matrix1(m1, row1, col1);
	Map<const MatrixXd> matrix2(m2, row2, col2);
	Map<MatrixXd> result(vout, row1, col2);
	result = matrix1 - matrix2;
}

// matrix product of m1 and m2.
EXPORT_API(void) dmult_(_In_ double* m1, const int row1, const int col1, _In_ double* m2, const int row2, const int col2, _Out_ double* vout)
{
	Map<const MatrixXd> matrix1(m1, row1, col1);
	Map<const MatrixXd> matrix2(m2, row2, col2);
	Map<MatrixXd> result(vout, row1, col2);
	result = matrix1 * matrix2;
}

// matrix product of m1 and v1.
EXPORT_API(void) dmultv_(_In_ double* m1, const int row1, const int col1, _In_ double* v1, const int length, _Out_ double* vout)
{
	Map<const MatrixXd> matrix(m1, row1, col1);
	Map<const VectorXd> vector(v1, length);
	Map<VectorXd> result(vout, row1);
	result = matrix * vector;
}

// matrix transpose.
EXPORT_API(void) dtransp_(_In_ double* m1, const int row1, const int col1, _Out_ double* vout)
{
	Map<const MatrixXd> matrix1(m1, row1, col1);
	Map<MatrixXd> result(vout, col1, row1);
	result = matrix1.transpose();
}

//  A * B^T
EXPORT_API(void) dmultt_(_In_ double* v1, const int row1, const int col1, _In_ double* v2, const int row2, const int col2, _Out_ double* vout)
{
	Map<const MatrixXd> matrix1(v1, row1, col1);
	Map<const MatrixXd> matrix2(v2, row2, col2);
	Map<MatrixXd> result(vout, row1, row2);
	result = matrix1 * matrix2.transpose();
}

//  A * A^T 
EXPORT_API(void) da_multt_(_In_ double* m1, const int row1, const int col1, _Out_ double* vout)
{
	Map<const MatrixXd> matrix1(m1, row1, col1);
	Map<MatrixXd> result(vout, row1, row1);
	result =  matrix1 * matrix1.transpose();
}

//  A^T * A
EXPORT_API(void) da_tmult_(_In_ double* m1, const int row1, const int col1, _Out_ double* vout)
{
	Map<const MatrixXd> matrix1(m1, row1, col1);
	Map<MatrixXd> result(vout, col1, col1);
	result = matrix1.transpose() * matrix1;
}

// matrix trace.
EXPORT_API(double) dtrace_(_In_ double* m1, const int row1, const int col1)
{
	Map<const MatrixXd> matrix1(m1, row1, col1);
	return matrix1.trace();
}


// matrix eigenvalues for general matrix.
EXPORT_API(void) deigenvalues_(_In_ double* m1, 
	const int size, 
	_Out_ double* out_real_eigen, 
	_Out_ double*  out_imag_eigen,
	_Out_ double* out_real_eigenvectors,
	_Out_ double* out_image_eigenvectors)
{
	Map<const MatrixXd> matrix(m1, size, size);
	EigenSolver<MatrixXd> esolver(matrix);
	VectorXcd  eigenvalues = esolver.eigenvalues();
	MatrixXcd  eigenvectors = esolver.eigenvectors();
	Map<VectorXd>  real_eigen(out_real_eigen, size);
	Map<VectorXd>  image_eigen(out_imag_eigen, size);
    real_eigen = eigenvalues.real();
	image_eigen = eigenvalues.imag();
	Map<MatrixXd>  real_eigen_vector(out_real_eigenvectors, size, size);
	Map<MatrixXd>  image_eigen_vector(out_image_eigenvectors, size, size);
	real_eigen_vector = eigenvectors.real();
	image_eigen_vector = eigenvectors.imag();
}


// matrix eigenvalues for self symetric matrix.
EXPORT_API(void) dselfadjoint_eigenvalues_(_In_ double* m1, const int size, _Out_ double* out_real_eigen, _Out_ double* out_real_eigenvectors)
{
	Map<const MatrixXd> matrix(m1, size, size);
	SelfAdjointEigenSolver<MatrixXd> esolver(matrix);
	VectorXcd  eigenvalues = esolver.eigenvalues();
	MatrixXcd  eigenvectors = esolver.eigenvectors();
	Map<VectorXd>  real_eigen(out_real_eigen, size);
	real_eigen = eigenvalues.real();
	Map<MatrixXd>  real_eigen_vector(out_real_eigenvectors, size, size);
	real_eigen_vector = eigenvectors.real();
}


// A = X + X^T
EXPORT_API(void) dxplusxt_(_In_ double* m1, int size, _Out_ double* vout)
{
	Map<const MatrixXd> matrix(m1, size, size);
	Map<MatrixXd> result(vout, size, size);
	result = matrix + matrix.transpose();
}

// A = X + Y
EXPORT_API(void) dxplusa_(_In_ double* v1, const int row1, const int col1, _In_ double* v2, const int row2, const int col2, _Out_ double* vout)
{
	Map<const MatrixXd> matrix1(v1, row1, col1);
	Map<const MatrixXd> matrix2(v2, row2, col2);
	Map<MatrixXd> result(vout, row1, row2);
	result = matrix1 + matrix2;
}

// svd
EXPORT_API(void) dsvd_(_In_ double* m1, const int row, const int col, _Out_ double* uout, _Out_ double* sout, _Out_ double* vout)
{
	int minRowsCols = MIN(row, col);
	Map<const MatrixXd> matrix1(m1, row, col);
	JacobiSVD<MatrixXd> svd(matrix1, ComputeThinU | ComputeThinV);
	Map<MatrixXd> u(uout, row, minRowsCols);
	Map<VectorXd> s(sout, minRowsCols);
	Map<MatrixXd> v(vout, col, minRowsCols);
	u = svd.matrixU();
	v = svd.matrixV();
	s = svd.singularValues();
}

EXPORT_API(void) dsvd_leastsquares_(_In_ double* m1, const int row, const int col, _In_ double* v1, _Out_ double* vout)
{
	int minRowsCols = MIN(row, col);
	Map<const MatrixXd> matrix1(m1, row, col);
	JacobiSVD<MatrixXd> svd(matrix1, ComputeThinU | ComputeThinV);
	Map<VectorXd> rhs(v1, row);
	Map<VectorXd> result(vout, minRowsCols);
	result = svd.solve(rhs);
}

EXPORT_API(void) dsvd_bdcSvd_(_In_ double* m1, const int row, const int col, _Out_ double* uout, _Out_ double* sout, _Out_ double* vout)
{
	int minRowsCols = MIN(row, col);
	Map<const MatrixXd> matrix1(m1, row, col);
	JacobiSVD<MatrixXd> bdcSvd(matrix1, ComputeThinU | ComputeThinV);
	Map<MatrixXd> u(uout, row, minRowsCols);
	Map<VectorXd> s(sout, minRowsCols);
	Map<MatrixXd> v(vout, col, minRowsCols);
	u = bdcSvd.matrixU();
	v = bdcSvd.matrixV();
	s = bdcSvd.singularValues();
}

EXPORT_API(void) dsvd_bdcSvd__leastsquares_(_In_ double* m1, const int row, const int col, _In_ double* v1, _Out_ double* vout)
{
	int minRowsCols = MIN(row, col);
	Map<const MatrixXd> matrix1(m1, row, col);
	JacobiSVD<MatrixXd> bdcSvd(matrix1, ComputeThinU | ComputeThinV);
	Map<VectorXd> rhs(v1, row);
	Map<VectorXd> result(vout, minRowsCols);
	result = bdcSvd.solve(rhs);
}

EXPORT_API(void) dnormal_equations__leastsquares_(_In_ double* m1, const int row, const int col, _In_ double* v1, _Out_ double* vout)
{
	Map<const MatrixXd> matrix1(m1, row, col);
	JacobiSVD<MatrixXd> bdcSvd(matrix1, ComputeThinU | ComputeThinV);
	Map<VectorXd> rhs(v1, row);
	Map<VectorXd> result(vout, col);
	result = (matrix1.transpose() * matrix1).ldlt().solve(matrix1.transpose() * rhs);
}

// Householder rank-revealing QR decomposition of a matrix with column-pivoting.
EXPORT_API(void) dsolve_colPivHouseholderQr_(_In_ double* m1, const int row, const int col, _In_ double* v1, _Out_ double* vout)
{
	Map<const MatrixXd> matrix1(m1, row, col);
	Map<VectorXd> rhs(v1, row);
	Map<VectorXd> result(vout, row);
	result = matrix1.colPivHouseholderQr().solve(rhs);
}

// Standard Cholesky decomposition (LL^T) of a matrix and associated features.
EXPORT_API(void) dsolve_LLT_(_In_ double* m1, const int row, const int col, _In_ double* v1, _Out_ double* vout)
{
	Map<const MatrixXd> matrix1(m1, row, col);
	Map<VectorXd> rhs(v1, row);
	Map<VectorXd> result(vout, row);
	result = matrix1.colPivHouseholderQr().solve(rhs);
}

EXPORT_API(double) ddeterminant_(_In_ double* m1, const int row, const int col)
{
	Map<const MatrixXd> matrix1(m1, row, col);
	return matrix1.determinant();
}

EXPORT_API(void) dinverse_(_In_ double* m1, const int row, const int col, _Out_ double* vout)
{
	Map<const MatrixXd> matrix1(m1, row, col);
	Map<MatrixXd> result(vout, row, row);
	result = matrix1.inverse();
}

EXPORT_API(double) drelative_error_(_In_ double* m1, const int row, const int col, _In_ double* v1, _In_ double* v2) {
	Map<const MatrixXd> matrix1(m1, row, col);
	Map<VectorXd> rhs(v1, row);
	Map<VectorXd> x(v2, row);
	return (matrix1 * x - rhs).norm() / rhs.norm();
}

EXPORT_API(double) dabsolute_error_(_In_ double* m1, const int row, const int col, _In_ double* v1, _In_ double* v2) {
	Map<const MatrixXd> matrix1(m1, row, col);
	Map<VectorXd> rhs(v1, row);
	Map<VectorXd> x(v2, row);
	return (matrix1 * x - rhs).norm();
}

EXPORT_API(void) dhouseholderQR_(_In_ double* m1, const int row, const int col, _Out_ double* v1, _Out_ double* v2) {
	Map<const MatrixXd> matrix1(m1, row, col);
	Map<MatrixXd> Q(v1, row, row);
	Map<MatrixXd> R(v2, row, col);
	HouseholderQR<MatrixXd> qr(matrix1);
    Q = qr.householderQ();
	R = qr.matrixQR().triangularView<Upper>();
}

EXPORT_API(void) dcolPivHouseholderQR_(_In_ double* m1, const int row, const int col, _Out_ double* v1, _Out_ double* v2, _Out_ double *v3) {
	Map<const MatrixXd> matrix1(m1, row, col);
	Map<MatrixXd> Q(v1, row, row);
	Map<MatrixXd> R(v2, row, col);
	Map<MatrixXd>  P(v3, col, col);
	ColPivHouseholderQR<MatrixXd> qr(matrix1);
	Q = qr.householderQ();
	R = qr.matrixQR().triangularView<Upper>();
    P = qr.colsPermutation();
}

EXPORT_API(void) dfullPivLU_(_In_ double* m1, 
	const int row, 
	const int col,
	_Out_ double* v1,
	_Out_ double* v2,
	_Out_ double* v3,
	_Out_ double* v4) {
	Map<const MatrixXd> matrix1(m1, row, col);
	Map<MatrixXd>  L(v1, row, row);
	Map<MatrixXd>  U(v2, row, col);
	Map<MatrixXd>  P(v3, row, row);
	Map<MatrixXd>  Q(v4, col, col);

	FullPivLU<MatrixXd> lu(matrix1);

	P = lu.permutationP();
	Q = lu.permutationQ();

	U = lu.matrixLU().triangularView<Upper>();
	L = lu.matrixLU().triangularView<StrictlyLower>();
}