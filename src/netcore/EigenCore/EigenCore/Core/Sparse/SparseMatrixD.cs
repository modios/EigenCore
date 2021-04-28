using EigenCore.Core.Dense;
using EigenCore.Core.Sparse.LinearAlgebra;
using EigenCore.Eigen;
using System;
using System.Collections.Generic;
using System.Linq;

namespace EigenCore.Core.Sparse
{
    public class SparseMatrixD : MatrixSparseBase<double>
    {
        private static readonly IterativeSolverInfo _defaultIterativeSolverInfo = new IterativeSolverInfo();

        private int NonZeroUpperBound(SparseMatrixD other)
        {
            var outerStarts = _outerStarts;
            var outStartsOther = other._outerStarts;
            var elementsPerColum = new int[Cols];
            var elementsPerColumOther = new int[other.Cols];
            for (int i = 0; i <= outerStarts.Length - 2; i++)
            {
                elementsPerColum[i] = outerStarts[i + 1] - outerStarts[i];
            }

            for (int i = 0; i <= outStartsOther.Length - 2; i++)
            {
                elementsPerColumOther[i] = outStartsOther[i + 1] - outStartsOther[i];
            }

            return Math.Min(elementsPerColum.Max() * elementsPerColumOther.Max() * Cols, Cols * Rows);
        }

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

        public static SparseMatrixD Random(int rows, int cols, double percentageNonZeros, double min = 0, double max = 1, int seed = 0)
        {

            double maxMinusMin = max - min;
            int nnz = (int)Math.Floor(rows * cols * percentageNonZeros);
            List<(int, int, double)> elements = new List<(int, int, double)>(nnz);
            HashSet<(int, int)> visitedPosition = new HashSet<(int, int)>();
            if (_random == null) SetRandomState(seed);

            while (elements.Count < nnz)
            {
                (int, int) position = (_random.Next(0, rows), _random.Next(0, cols));
                if (visitedPosition.Add(position)) {
                    elements.Add((position.Item1, position.Item2, maxMinusMin * _random.NextDouble() + min));
                }
            }

            return new SparseMatrixD(elements, rows, cols);
        }

        public static SparseMatrixD Identity(int size)
        {
            (int, int, double)[] positions = new (int, int, double)[size];

            for (int i = 0; i < size; i++)
            {
                positions[i] = (i, i, 1.0);
            }

            return new SparseMatrixD(positions, size, size);
        }

        public static SparseMatrixD Diag(double[] values)
        {
            var size = values.Length;
            (int, int, double)[] positions = new (int, int, double)[size];

            for (int i = 0; i < size; i++)
            {
                positions[i] = (i, i, values[i]);
            }

            return new SparseMatrixD(positions, size, size);
        }

        public double Max() => _values.AsParallel().Max();

        public double Min() => _values.AsParallel().Min();

        public double Sum() => _values.AsParallel().Sum();

        public double Prod() => _values.AsParallel().Aggregate((product, nextElement) => product * nextElement);

        public double Mean() => _values.AsParallel().Average();

        public double Norm() => EigenSparseUtilities.Norm(Rows, Cols,Nnz, GetOuterStarts(), GetInnerIndices(), GetValues());

        public double SquaredNorm() => EigenSparseUtilities.SquaredNorm(Rows, Cols, Nnz, GetOuterStarts(), GetInnerIndices(), GetValues());

        public void Scale(double scalar)
        {
            for (int i = 0; i < Nnz; i++)
            {
                _values[i] = _values[i] * scalar;
            }
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

        public SparseMatrixD Add(SparseMatrixD other)
        {
            int[] innerIndices = new int[Nnz + other.Nnz];
            int[] outOuterStarts = new int[Cols + 1];
            double[] values = new double[Nnz + other.Nnz];
            int nnz;
            Eigen.EigenSparseUtilities.ADD(Rows, Cols,
               Nnz, GetOuterStarts(), GetInnerIndices(), GetValues(),
               other.Nnz, other.GetOuterStarts(), other.GetInnerIndices(), other.GetValues(),
               outOuterStarts, innerIndices, values, out nnz);
            Array.Resize(ref innerIndices, nnz);
            Array.Resize(ref values, nnz);
            return new SparseMatrixD(values, innerIndices, outOuterStarts, Rows, Cols);
        }

        public SparseMatrixD Minus(SparseMatrixD other)
        {
            int[] innerIndices = new int[Nnz + other.Nnz];
            int[] outOuterStarts = new int[Cols + 1];
            double[] values = new double[Nnz + other.Nnz];
            int nnz;
            Eigen.EigenSparseUtilities.Minus(Rows, Cols,
               Nnz, GetOuterStarts(), GetInnerIndices(), GetValues(),
               other.Nnz, other.GetOuterStarts(), other.GetInnerIndices(), other.GetValues(),
               outOuterStarts, innerIndices, values, out nnz);
            Array.Resize(ref innerIndices, nnz);
            Array.Resize(ref values, nnz);
            return new SparseMatrixD(values, innerIndices, outOuterStarts, Rows, Cols);
        }

        public VectorXD Mult(VectorXD other)
        {
            double[] values = new double[Rows];
            Eigen.EigenSparseUtilities.Mult(Rows, Cols,
               Nnz, GetOuterStarts(), GetInnerIndices(), GetValues(), other.GetValues(), other.Length, values);
            return new VectorXD(values);
        }

        public SparseMatrixD Mult(SparseMatrixD other)
        {
            var upperBound = NonZeroUpperBound(other);
            int[] innerIndices = new int[upperBound];
            int[] outOuterStarts = new int[Cols + 1];
            double[] values = new double[upperBound];
            int nnz;
            Eigen.EigenSparseUtilities.Mult(Rows, Cols,
               Nnz, GetOuterStarts(), GetInnerIndices(), GetValues(),
               other.Nnz, other.GetOuterStarts(), other.GetInnerIndices(), other.GetValues(),
               outOuterStarts, innerIndices, values, out nnz);
            Array.Resize(ref innerIndices, nnz);
            Array.Resize(ref values, nnz);
            return new SparseMatrixD(values, innerIndices, outOuterStarts, Rows, Cols);
        }

        public SparseMatrixD Transpose()
        {
            int[] innerIndices = new int[Nnz];
            int[] outOuterStarts = new int[Rows + 1];
            double[] values = new double[Nnz];
            Eigen.EigenSparseUtilities.Transpose(Rows, Cols,
               Nnz, GetOuterStarts(), GetInnerIndices(), GetValues(),
               outOuterStarts, innerIndices, values);

            return new SparseMatrixD(values, innerIndices, outOuterStarts, Cols, Rows);
        }

        public IterativeSolverResult IterativeSolve(VectorXD other, IterativeSolverInfo iterativeSolverInfo = default(IterativeSolverInfo))
        {
            double[] x = new double[other.Length];
            bool success;
            double error;
            int iterations;

            if(iterativeSolverInfo == default(IterativeSolverInfo))
            {
                iterativeSolverInfo = _defaultIterativeSolverInfo;
            }

            switch (iterativeSolverInfo.Solver)
            {
                case IterativeSolverType.BiCGSTAB:
                    success = EigenSparseUtilities.SolveBiCGSTAB(
                       Rows,
                       Cols,
                       Nnz,
                       iterativeSolverInfo.MaxIterations,
                       iterativeSolverInfo.Tolerance,
                       GetOuterStarts(),
                       GetInnerIndices(),
                       GetValues(),
                       other.GetValues(),
                       other.Length,
                       x,
                       out iterations,
                       out error);
                    break;
                case IterativeSolverType.GMRES:
                    success = EigenSparseUtilities.SolveGMRES(
                       Rows,
                       Cols,
                       Nnz,
                       iterativeSolverInfo.MaxIterations,
                       iterativeSolverInfo.Tolerance,
                       GetOuterStarts(),
                       GetInnerIndices(),
                       GetValues(),
                       other.GetValues(),
                       other.Length,
                       x,
                       out iterations,
                       out error);
                    break;
                case IterativeSolverType.MINRES:
                    success = EigenSparseUtilities.SolveMINRES(
                       Rows,
                       Cols,
                       Nnz,
                       iterativeSolverInfo.MaxIterations,
                       iterativeSolverInfo.Tolerance,
                       GetOuterStarts(),
                       GetInnerIndices(),
                       GetValues(),
                       other.GetValues(),
                       other.Length,
                       x,
                       out iterations,
                       out error);
                    break;
                case IterativeSolverType.DGMRES:
                    success = EigenSparseUtilities.SolveDGMRES(
                       Rows,
                       Cols,
                       Nnz,
                       iterativeSolverInfo.MaxIterations,
                       iterativeSolverInfo.Tolerance,
                       GetOuterStarts(),
                       GetInnerIndices(),
                       GetValues(),
                       other.GetValues(),
                       other.Length,
                       x,
                       out iterations,
                       out error);
                    break;
                case IterativeSolverType.LeastSquaresConjugateGradient:
                    success = EigenSparseUtilities.SolveLeastSquaresConjugateGradient(
                       Rows,
                       Cols,
                       Nnz,
                       iterativeSolverInfo.MaxIterations,
                       iterativeSolverInfo.Tolerance,
                       GetOuterStarts(),
                       GetInnerIndices(),
                       GetValues(),
                       other.GetValues(),
                       other.Length,
                       x,
                       out iterations,
                       out error);
                    break;
                case IterativeSolverType.ConjugateGradient:
                default:
                    success = EigenSparseUtilities.SolveConjugateGradient(
                       Rows,
                       Cols,
                       Nnz,
                       iterativeSolverInfo.MaxIterations,
                       iterativeSolverInfo.Tolerance,
                       GetOuterStarts(),
                       GetInnerIndices(),
                       GetValues(),
                       other.GetValues(),
                       other.Length,
                       x,
                       out iterations,
                       out error);
                    break;
            }

            return new IterativeSolverResult(new VectorXD(x), iterations, error, iterativeSolverInfo.Solver, success);
        }

        public VectorXD DirectSolve(VectorXD other, DirectSolverType directSolverType = DirectSolverType.SparseLU)
        {
            double[] x = new double[other.Length];
            switch (directSolverType)
            {
                case DirectSolverType.SimplicialLLT:
                    EigenSparseUtilities.SolveSimplicialLLT(Rows, Cols, Nnz, GetOuterStarts(),
                        GetInnerIndices(), GetValues(), other.GetValues(), other.Length, x);
                    break;
                case DirectSolverType.SimplicialLDLT:
                    EigenSparseUtilities.SolveSimplicialLDLT(Rows, Cols, Nnz, GetOuterStarts(),
                      GetInnerIndices(), GetValues(), other.GetValues(), other.Length, x);
                    break;
                case DirectSolverType.SparseQR:
                    EigenSparseUtilities.SolveSparseQR(Rows, Cols, Nnz, GetOuterStarts(),
                      GetInnerIndices(), GetValues(), other.GetValues(), other.Length, x);
                    break;
                case DirectSolverType.SparseLU:
                default:
                    EigenSparseUtilities.SolveSparseLU(Rows, Cols, Nnz, GetOuterStarts(),
                      GetInnerIndices(), GetValues(), other.GetValues(), other.Length, x);
                    break;
            }

            return new VectorXD(x);
        }

        public VectorXD LeastSquares(VectorXD other)
        {
            double[] x = new double[Cols];
            EigenSparseUtilities.LeastSquaresLU(Rows, Cols, Nnz, GetOuterStarts(),
                GetInnerIndices(), GetValues(), other.GetValues(), other.Length, x);

            return new VectorXD(x);
        }

        public SparseMatrixD(IList<(int, int, double)> sparseInfo, int rows, int cols)
            : base(MatrixSparseHelpers.ToCCS(sparseInfo.ToList(), cols), rows, cols)
        {
        }

        public SparseMatrixD(double[] values, int[] innerIndices, int[] outerStarts, int rows, int cols) 
            : base(values, innerIndices, outerStarts, rows, cols)
        {
        }
    }
}
