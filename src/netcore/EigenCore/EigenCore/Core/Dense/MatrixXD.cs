using EigenCore.Core.Dense.Complex;
using EigenCore.Core.Dense.LinearAlgebra;
using EigenCore.Eigen;
using System.Linq;

namespace EigenCore.Core.Dense
{
    public class MatrixXD : MatrixDenseBase<double>
    {
        private bool IsEqual(MatrixXD other)
        {
            if (Rows != other.Rows || Cols != other.Cols)
            {
                return false;
            }

            return VectorHelpers.ArraysEqual(_values, other.GetValues().ToArray());
        }

        public static MatrixXD Zeros(int rows, int cols)
        {
            return new MatrixXD(new double[rows * cols], rows, cols);
        }

        public static MatrixXD Ones(int rows, int cols)
        {
            var values = new double[rows * cols];
            values.Populate(1.0);
            return new MatrixXD(values, rows, cols);
        }

        public static MatrixXD Random(int rows, int cols, double min = 0, double max = 1, int seed = 0)
        {
            double[] input = new double[rows * cols];
            double maxMinusMin = max - min;

            if (_random == null) SetRandomState(seed);

            for (int i = 0; i < rows * cols; i++)
            {
                input[i] = maxMinusMin * _random.NextDouble() + min;
            }

            return new MatrixXD(input, rows, cols);
        }

        public static MatrixXD Identity(int size)
        {
            double[] input = new double[size * size];

            for (int i = 0; i < size; i++)
            {
                input[i * (size + 1)] = 1.0;
            }

            return new MatrixXD(input, size, size);
        }

        public static MatrixXD Diag(double[] values)
        {
            var size = values.Length;
            double[] input = new double[size * size];

            for (int i = 0; i < size; i++)
            {
                input[i * (size + 1)] = values[i];
            }

            return new MatrixXD(input, size, size);
        }

        public double Max() => _values.AsParallel().Max();

        public double Min() => _values.AsParallel().Min();

        public double Sum() => _values.AsParallel().Sum();

        public double Prod() => _values.AsParallel().Aggregate((product, nextElement) => product * nextElement);

        public double Mean() => _values.AsParallel().Average();

        public double Norm()
        {
            return EigenDenseUtilities.Norm(GetValues(), Rows, Cols);
        }

        public double SquaredNorm()
        {
            return EigenDenseUtilities.SquaredNorm(GetValues(), Rows, Cols);
        }

        public double Lp1Norm()
        {
            return EigenDenseUtilities.Lp1Norm(GetValues(), Rows, Cols);
        }

        public double LpInfNorm()
        {
            return EigenDenseUtilities.LpInfNorm(GetValues(), Rows, Cols);
        }

        public VectorXD ColwiseMin()
        {
            double[] result = new double[Cols];

            for(int i=0; i< Cols; i++)
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

        public void Scale(double scalar)
        {
            for (int i = 0; i < Length; i++)
            {
                _values[i] = _values[i] * scalar;
            }
        }

        public VectorXD Row(int row)
        {
            double[] rowVector = new double[Cols];

            for(int i=0; i<Cols; i++)
            {
                rowVector[i] = Get(row, i);
            }

            return new VectorXD(rowVector);
        }

        public VectorXD Col(int col)
        {
            double[] rowVector = new double[Rows];

            for (int i = 0; i < Rows; i++)
            {
                rowVector[i] = Get(i, col);
            }

            return new VectorXD(rowVector);
        }

        public MatrixXD Slice(int[] rows, int[] cols)
        {
            int nrows = rows.Length;
            int ncols = cols.Length;
            double[] inputValues = new double[nrows * ncols];
            MatrixXD matrixXD = new MatrixXD(inputValues, nrows, ncols);

            for (int i = 0; i < nrows; i++)
            {
                for (int j = 0; j < ncols; j++)
                {
                    matrixXD.Set(i, j, Get(rows[i], cols[j]));
                }
            }

            return matrixXD;
        }


        public MatrixXD Concat(MatrixXD other, ConcatType concatType)
        {
            switch (concatType)
            {
                case ConcatType.Horizontal:
                    {
                        var totalCols = Cols + other.Cols;

                        double[] inputValues = new double[Rows * totalCols];
                        var matrixXD = new MatrixXD(inputValues, Rows, totalCols);

                        for (int i = 0; i < Rows; i++)
                        {
                            for (int j = 0; j < Cols; j++)
                            {
                                matrixXD.Set(i, j, Get(i, j));
                            }
                        }

                        int otherCols = other.Cols;
                        for (int i = 0; i < other.Rows; i++)
                        {
                            for (int j = 0 ; j < otherCols; j++)
                            {
                                matrixXD.Set(i, Cols + j, other.Get(i, j));
                            }
                        }

                        return matrixXD;
                    }
                case ConcatType.Vertical:
                default:
                    {

                        int totalRows = Rows + other.Rows;

                        double[] inputValues = new double[totalRows * Cols];
                        MatrixXD matrixXD = new MatrixXD(inputValues, totalRows, Cols);

                        for (int i = 0; i < Rows; i++)
                        {
                            for (int j = 0; j < Cols; j++)
                            {
                                matrixXD.Set(i, j, Get(i, j));
                            }
                        }

                        int otherRows= other.Rows;
                        for (int i = 0; i < otherRows; i++)
                        {
                            for (int j = 0; j < Cols; j++)
                            {
                                matrixXD.Set(Rows + i, j, other.Get(i, j));
                            }
                        }

                        return matrixXD;
                    }
            }
        }

        public MatrixXD Slice(int startRow, int endRow, int startCol, int endCol)
        {
            var rows = Enumerable.Range(startRow, endRow - startRow  + 1).ToArray();
            var cols = Enumerable.Range(startCol, endCol- startCol + 1).ToArray();

            return Slice(rows, cols);
        }

        public void SetDiag(double scalar)
        {
            for (int i = 0; i < Cols; i++)
            {
                Set(i, i, scalar);
            }
        }      
        
        public MatrixXD Minus(MatrixXD other)
        {
            double[] outMatrix = new double[Rows * other.Cols];
            EigenDenseUtilities.Minus(GetValues(),
                Rows,
                Cols,
                other.GetValues(),
                other.Rows,
                other.Cols,
                outMatrix);
            return new MatrixXD(outMatrix, Rows, other.Cols);
        }

        public MatrixXD Mult(MatrixXD other)
        {
            double[] outMatrix = new double[Rows * other.Cols];
            EigenDenseUtilities.Mult(GetValues(),
                Rows,
                Cols,
                other.GetValues(),
                other.Rows,
                other.Cols,
                outMatrix);
            return new MatrixXD(outMatrix, Rows, other.Cols);
        }

        public VectorXD Mult(VectorXD other)
        {
            double[] outVector = new double[Rows];
            EigenDenseUtilities.Mult(GetValues(),
                Rows,
                Cols,
                other.GetValues(),
                other.Length,
                outVector);
            return new VectorXD(outVector);
        }

        public MatrixXD Plus(MatrixXD other)
        {
            double[] outMatrix = new double[Rows * Cols];
            EigenDenseUtilities.Plus(
                GetValues(),
                Rows,
                Cols,
                other.GetValues(),
                other.Rows,
                other.Cols,
                outMatrix);
            return new MatrixXD(outMatrix, Rows, Cols);
        }

        public MatrixXD Transpose()
        {
            double[] outMatrix = new double[Rows * Cols];
            EigenDenseUtilities.Transpose(GetValues(), Rows, Cols, outMatrix);
            return new MatrixXD(outMatrix, Cols, Rows);
        }

        // A * B^T
        public MatrixXD MultT(MatrixXD other)
        {
            double[] outMatrix = new double[Rows * other.Rows];
            EigenDenseUtilities.MultT(GetValues(), Rows, Cols, other.GetValues(), other.Rows, other.Cols, outMatrix);
            return new MatrixXD(outMatrix, Rows, other.Rows);
        }

        public double Trace()
        {
            return EigenDenseUtilities.Trace(GetValues(), Rows, Cols);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public VectorXD Solve(VectorXD other, DenseSolverType denseSolverType = DenseSolverType.ColPivHouseholderQR)
        {
            double[] vout = new double[Rows];
            switch (denseSolverType)
            {
                case DenseSolverType.LLT:
                    EigenDenseUtilities.SolveLLT(GetValues(), Rows, Cols, other.GetValues(), vout);
                    break;
                case DenseSolverType.ColPivHouseholderQR:
                default:
                    EigenDenseUtilities.SolveColPivHouseholderQr(GetValues(), Rows, Cols, other.GetValues(), vout);
                    break;
            }

            return new VectorXD(vout);
        }

        public double Determinant()
        {
            return EigenDenseUtilities.Determinant(GetValues(), Rows, Cols);
        }

        public MatrixXD Inverse()
        {
            double[] vout = new double[Rows * Cols];
            EigenDenseUtilities.Determinant(GetValues(), Rows, Cols, vout);
            return new MatrixXD(vout, Rows, Cols);
        }

        public double AbsoluteError(VectorXD rhs, VectorXD x)
        {
            return EigenDenseUtilities.AbsoluteError(GetValues(), Rows, Cols, rhs.GetValues(), x.GetValues());
        }

        public double RelativeError(VectorXD rhs, VectorXD x)
        {
            return EigenDenseUtilities.RelativeError(GetValues(), Rows, Cols, rhs.GetValues(), x.GetValues());
        }

        /// <summary>
        /// eigenvalues and eigenvectros.
        /// </summary>
        /// <returns></returns>
        public EigenSolverResult Eigen()
        {
            double[] realValues = new double[Rows];
            double[] imagValues = new double[Rows];
            double[] realEigenvectors = new double[Rows * Cols];
            double[] imagEigenvectors = new double[Rows * Cols];

            EigenDenseUtilities.EigenSolver(GetValues(), Rows, realValues, imagValues, realEigenvectors, imagEigenvectors);

            return new EigenSolverResult(new VectorXCD(realValues, imagValues), new MatrixXCD(realEigenvectors, imagEigenvectors, Rows, Cols));
        }

        public SVDResult SVD(SVDType svdType = SVDType.Jacobi)
        {
            int minRowsCols = Cols < Rows ? Cols : Rows;
            double[] uout = new double[Rows * minRowsCols];
            double[] sout = new double[minRowsCols];
            double[] vout = new double[Cols * minRowsCols];

            if (svdType == SVDType.Jacobi)
            {
                EigenDenseUtilities.SVD(GetValues(), Rows, Cols, uout, sout, vout);
            }
            else
            {
                EigenDenseUtilities.SVDBdcSvd(GetValues(), Rows, Cols, uout, sout, vout);
            }

            return new SVDResult(new MatrixXD(uout, Rows, minRowsCols),
                new VectorXD(sout),
                new MatrixXD(vout, Cols, minRowsCols));
        }

        /// <summary>
        /// X = A + A^T;
        /// </summary>
        public MatrixXD PlusT()
        {
            double[] outMatrix = new double[Rows * Cols];
            EigenDenseUtilities.PlusT(GetValues(), Rows, outMatrix);
            return new MatrixXD(outMatrix, Rows, Cols);
        }

        /// <summary>
        /// X = A * A^T;
        /// </summary>
        public MatrixXD MultT()
        {
            double[] outMatrix = new double[Rows * Rows];
            EigenDenseUtilities.MultT(GetValues(), Rows, Rows, outMatrix);
            return new MatrixXD(outMatrix, Rows, Rows);
        }

        /// <summary>
        /// X = A^T * A;
        /// </summary>
        public MatrixXD TMult()
        {
            double[] outMatrix = new double[Cols * Cols];
            EigenDenseUtilities.TMult(GetValues(), Rows, Cols, outMatrix);
            return new MatrixXD(outMatrix, Cols, Cols);
        }

        /// <summary>
        /// eigenvalues and eigenvectors for symmetric matrix.
        /// </summary>
        /// <returns></returns>
        public SAEigenSolverResult SymmetricEigen()
        {
            double[] realValues = new double[Rows];
            double[] realEigenvectors = new double[Rows * Cols];

            EigenDenseUtilities.SelfAdjointEigenSolver(GetValues(), Rows, realValues, realEigenvectors);

            return new SAEigenSolverResult(new VectorXD(realValues), new MatrixXD(realEigenvectors, Rows, Cols));
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="rhs"></param>
        public VectorXD LeastSquaresSVD(VectorXD rhs, SVDType svdType = SVDType.Jacobi)
        {
            double[] vout = new double[Cols];
            if (svdType == SVDType.Jacobi)
            {
                EigenDenseUtilities.SVDLeastSquares(GetValues(), Rows, Cols, rhs.GetValues(), vout);
            }
            else
            {
               EigenDenseUtilities.SVDLeastSquaresBdcSvd(GetValues(), Rows, Cols, rhs.GetValues(), vout);
            }

            return new VectorXD(vout);
        }

        /// <summary>
        /// A^TAx = A^b
        /// </summary>
        /// <param name="rhs"></param>
        /// <returns></returns>
        public VectorXD LeastSquaresNE(VectorXD rhs)
        {
            double[] vout = new double[Cols];
            EigenDenseUtilities.NormalEquationsLeastSquares(GetValues(), Rows, Cols, rhs.GetValues(), vout);
            return new VectorXD(vout);
        }

        /// <summary>
        /// QR decomposition
        /// </summary>
        /// <param name="qRType"></param>
        /// <returns></returns>
        public QRResult QR(QRType qRType = QRType.HouseholderQR)
        {
            double[] q = new double[Rows * Rows];
            double[] r = new double[Rows * Cols];

            switch (qRType)
            {
                case QRType.ColPivHouseholderQR:
                    double[] p = new double[Cols * Cols];
                    EigenDenseUtilities.ColPivHouseholderQR(GetValues(), Rows, Cols, q, r, p);
                    return new QRResult(new MatrixXD(q, Rows, Rows), new MatrixXD(r, Rows, Cols), new MatrixXD(p, Cols, Cols));
                case QRType.HouseholderQR:
                default:
                    EigenDenseUtilities.HouseholderQR(GetValues(), Rows, Cols, q, r);
                    break;
            }

            return new QRResult(new MatrixXD(q, Rows, Rows), new MatrixXD(r, Rows, Cols));
        }

        public FullPivLUResult FullPivLU()
        {
            double[] l = new double[Rows * Rows];
            double[] u = new double[Rows * Cols];
            double[] p = new double[Rows * Rows];
            double[] q = new double[Cols * Cols];

            EigenDenseUtilities.FullPivLU(GetValues(),Rows, Cols, l, u, p, q);

            var L = new MatrixXD(l, Rows, Rows);
            L.SetDiag(1.0);
            return new FullPivLUResult(L,
                new MatrixXD(u, Rows, Cols),
                new MatrixXD(p, Rows, Rows),
                new MatrixXD(q, Cols, Cols));

        }

        public override MatrixXD Clone()
        {
            return new MatrixXD(_values.ToArray(), Rows, Cols);
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

            return IsEqual((MatrixXD)value);
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        internal MatrixXD(double[] values, int rows, int cols)
                : base(values, rows, cols)
        {
        }

        public MatrixXD(double[][] inputValues) :
            base(() => JaggedToFlatColumnWise(inputValues),
            JaggedRowsAndColsInfo(inputValues).Item1,
            JaggedRowsAndColsInfo(inputValues).Item2)
        {
 
        }

        public MatrixXD(double[,] inputValues) :
            base(() => MultyDimToFlatColumnWise(inputValues),
            MultDimRowsAndColsInfo(inputValues).Item1,
            MultDimRowsAndColsInfo(inputValues).Item2)
        {

        }

        public MatrixXD(MatrixXD matrixXD)
        : base(matrixXD._values.ToArray(), matrixXD.Rows, matrixXD.Cols)
        {
        }

        public MatrixXD(string valuesString)
            : base(valuesString, (string value) => double.Parse(value))
        {

        }
    }
}
