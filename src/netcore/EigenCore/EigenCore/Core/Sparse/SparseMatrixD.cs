using EigenCore.Core.Dense;
using EigenCore.Core.Sparse.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;

namespace EigenCore.Core.Sparse
{
    public class SparseMatrixD : MatrixSparseBase<double>
    {
        private static readonly IterativeSolverInfo _defaultIterativeSolverInfo = new IterativeSolverInfo();
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
            double[] valeus = new double[Nnz + other.Nnz];
            int nnz;
            Eigen.EigenSparseUtilities.ADD(Rows, Cols,
               Nnz, GetOuterStarts(), GetInnerIndices(), GetValues(),
               other.Nnz, other.GetOuterStarts(), other.GetInnerIndices(), other.GetValues(),
               outOuterStarts, innerIndices, valeus, out nnz);
            Array.Resize(ref innerIndices, nnz);
            Array.Resize(ref valeus, nnz);
            return new SparseMatrixD(valeus, innerIndices, outOuterStarts, Rows, Cols);
        }

        public SparseMatrixD Minus(SparseMatrixD other)
        {
            int[] innerIndices = new int[Nnz + other.Nnz];
            int[] outOuterStarts = new int[Cols + 1];
            double[] valeus = new double[Nnz + other.Nnz];
            int nnz;
            Eigen.EigenSparseUtilities.Minus(Rows, Cols,
               Nnz, GetOuterStarts(), GetInnerIndices(), GetValues(),
               other.Nnz, other.GetOuterStarts(), other.GetInnerIndices(), other.GetValues(),
               outOuterStarts, innerIndices, valeus, out nnz);
            Array.Resize(ref innerIndices, nnz);
            Array.Resize(ref valeus, nnz);
            return new SparseMatrixD(valeus, innerIndices, outOuterStarts, Rows, Cols);
        }

        public IterativeSolverResult Solve(VectorXD other, IterativeSolverInfo iterativeSolverInfo = default(IterativeSolverInfo))
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
                    success = Eigen.EigenSparseUtilities.SolveBiCGSTAB(
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
                    success = Eigen.EigenSparseUtilities.SolveConjugateGradient(
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

            return success ? new IterativeSolverResult(new VectorXD(x), iterations, error, iterativeSolverInfo.Solver) 
                : default(IterativeSolverResult);
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
