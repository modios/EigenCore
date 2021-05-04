using EigenCore.Core.Dense;
using EigenCore.Core.Shared;
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

            return ArrayHelpers.ArraysEqual(_values, other._values) &&
                ArrayHelpers.ArraysEqual(_innerIndices, other._innerIndices) &&
                ArrayHelpers.ArraysEqual(_outerStarts, other._outerStarts);
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

        public SparseVectorD Row(int row)
        {
            var indxAndRow = GetRow(row);
            int[] indices = indxAndRow.Item1;
            double[] values = indxAndRow.Item2;
            return new SparseVectorD(indices, values, Cols);
        }

        public SparseVectorD Col(int col)
        {
            var indxAndRow = GetCol(col);
            int[] indices = indxAndRow.Item1;
            double[] values = indxAndRow.Item2;
            return new SparseVectorD(indices, values, Rows);
        }

        public VectorXD ColwiseMin()
        {
            double[] result = new double[Cols];

            for (int i = 0; i < Cols; i++)
            {
                result[i] = Col(i).Min();
            }

            return new VectorXD(result);
        }

        public VectorXD RowwiseMin()
        {
            double[] result = new double[Rows];

            for (int i = 0; i < Rows; i++)
            {
                result[i] = Row(i).Min();
            }

            return new VectorXD(result);
        }

        public VectorXD ColwiseMax()
        {
            double[] result = new double[Cols];

            for (int i = 0; i < Cols; i++)
            {
                result[i] = Col(i).Max();
            }

            return new VectorXD(result);
        }

        public VectorXD RowwiseMax()
        {
            double[] result = new double[Rows];

            for (int i = 0; i < Rows; i++)
            {
                result[i] = Row(i).Max();
            }

            return new VectorXD(result);
        }

        public VectorXD ColwiseSum()
        {
            double[] result = new double[Cols];

            for (int i = 0; i < Cols; i++)
            {
                result[i] = Col(i).Sum();
            }

            return new VectorXD(result);
        }

        public VectorXD RowwiseSum()
        {
            double[] result = new double[Rows];

            for (int i = 0; i < Rows; i++)
            {
                result[i] = Row(i).Sum();
            }

            return new VectorXD(result);
        }

        public VectorXD ColwiseProd()
        {
            double[] result = new double[Cols];

            for (int i = 0; i < Cols; i++)
            {
                result[i] = Col(i).Prod();
            }

            return new VectorXD(result);
        }

        public VectorXD RowwiseProd()
        {
            double[] result = new double[Rows];

            for (int i = 0; i < Rows; i++)
            {
                result[i] = Row(i).Prod();
            }

            return new VectorXD(result);
        }

        public VectorXD ColwiseMean()
        {
            double[] result = new double[Cols];

            for (int i = 0; i < Cols; i++)
            {
                result[i] = Col(i).Mean();
            }

            return new VectorXD(result);
        }

        public VectorXD RowwiseMean()
        {
            double[] result = new double[Rows];

            for (int i = 0; i < Rows; i++)
            {
                result[i] = Row(i).Mean();
            }

            return new VectorXD(result);
        }

        public SparseMatrixD Concat(SparseMatrixD other, ConcatType concatType)
        {
            switch (concatType)
            {
                case ConcatType.Vertical:
                    var valuesV = new double[_values.Length + other._values.Length];
                    var outerV = ArrayHelpers.SumArrays(_outerStarts, other._outerStarts);
                    var innerV = new int[valuesV.Length];
                    int colIndex = 0;

                    for (int i = 0; i < other._outerStarts.Length - 1; i++)
                    {
                        int startColIndex = _outerStarts[i];
                        int colElements = _outerStarts[i + 1] - _outerStarts[i];

                        int startColIndexOther = other._outerStarts[i];
                        int colElementsOther = other._outerStarts[i + 1] - other._outerStarts[i];

                        var tmpInnner = new int[colElementsOther];
                        for (int j = 0; j < colElementsOther; j++)
                        {
                            tmpInnner[j] = other._innerIndices[startColIndexOther + j] + Rows;
                        }

                        Array.Copy(_innerIndices, startColIndex, innerV, colIndex, colElements);
                        Array.Copy(_values, startColIndex, valuesV, colIndex, colElements);
                        colIndex += colElements;
                        Array.Copy(tmpInnner, 0, innerV, colIndex, colElementsOther);
                        Array.Copy(other._values, startColIndexOther, valuesV, colIndex, colElementsOther);
                        colIndex += colElementsOther;
                    }

                    return new SparseMatrixD(valuesV, innerV, outerV, Rows + other.Rows, Cols);
                case ConcatType.Horizontal:
                default:
                    int[] innerH = _innerIndices.Concat(other._innerIndices).ToArray();
                    var outerLastH = _outerStarts.Last();
                    var lengthH = _outerStarts.Length;
                    int[] outerH = new int[_outerStarts.Length + other._outerStarts.Length - 1];

                    for (int i = 0; i < other._outerStarts.Length - 1; i++)
                    {
                        outerH[i + lengthH] = other._outerStarts[i + 1] + outerLastH;
                    }

                    Array.Copy(_outerStarts, outerH, _outerStarts.Length);
                    var values = _values.Concat(other._values).ToArray();
                    return new SparseMatrixD(values, innerH, outerH, Rows, Cols + other.Cols);
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

        public double AbsoluteError(VectorXD rhs, VectorXD x)
        {
            return EigenSparseUtilities.AbsoluteError(Rows, Cols, Nnz, GetOuterStarts(),
                      GetInnerIndices(), GetValues(), rhs.GetValues(), x.GetValues());
        }

        public double RelativeError(VectorXD rhs, VectorXD x)
        {
            return EigenSparseUtilities.RelativeError(Rows, Cols, Nnz, GetOuterStarts(),
                      GetInnerIndices(), GetValues(), rhs.GetValues(), x.GetValues());
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
