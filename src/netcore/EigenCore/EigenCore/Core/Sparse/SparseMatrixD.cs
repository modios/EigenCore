using EigenCore.Core.Dense;
using EigenCore.Core.Sparse.LinearAlgebra;
using System.Collections.Generic;

namespace EigenCore.Core.Sparse
{
    public class SparseMatrixD : MatrixSparseBase<double>
    {
        private bool IsEqual(SparseMatrixD other)
        {
            if (Rows != other.Rows || Cols != other.Cols)
            {
                return false;
            }

            return VectorHelpers.ArraysEqual(_values, other._values) &&
                VectorHelpers.ArraysEqual(_innerIndices, other._innerIndices) &&
                VectorHelpers.ArraysEqual(_outerStarts, other._outerStarts);
        }

        public override bool Equals(object value)
        {
            if (ReferenceEquals(null, value))
            {
                return false;
            }

            if (ReferenceEquals(this, value))
            {
                return true;
            }

            if (value.GetType() != GetType())
            {
                return false;
            }

            return IsEqual((SparseMatrixD)value);
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        public VectorXD Solve(VectorXD other, SparseSolveType sparseSolveType = SparseSolveType.ConjugateGradient)
        {
            double[] x = new double[other.Length];
            bool success;
            switch (sparseSolveType)
            {
                case SparseSolveType.ConjugateGradient:
                default:
                    success = Eigen.EigenSparseUtilities.ConjugateGradient(
                       Rows,
                       Cols,
                       Nnz,
                       GetOuterStarts(),
                       GetInnerIndices(),
                       GetValues(),
                       other.GetValues(),
                       other.Length,
                       x);
                    break;
            }

            return success ? new VectorXD(x) : default(VectorXD);
        }

        public SparseMatrixD(IList<(int, int, double)> sparseInfo, int rows, int cols)
            : base(MatrixSparseHelpers.ToCCS(sparseInfo, cols), rows, cols)
        {
        }

        public SparseMatrixD(double[] values, int[] innerIndices, int[] outerStarts, int rows, int cols) 
            : base(values, innerIndices, outerStarts, rows, cols)
        {
        }
    }
}
