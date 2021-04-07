// EigenNative.cpp : Defines the entry point for the application.

#include "EigenNative.h"
#include <Eigen/Core>

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

// matrix product of m1 and m2.
EXPORT_API(void) dmult_(_In_ double* m1, const int row1, const int col1, _In_ double* m2, const int row2, const int col2, _Out_ double* vout)
{
	Map<const MatrixXd> matrix1(m1, row1, col1);
	Map<const MatrixXd> matrix2(m2, row2, col2);
	Map<MatrixXd> result(vout, row1, col2);
	result = matrix1 * matrix2;
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