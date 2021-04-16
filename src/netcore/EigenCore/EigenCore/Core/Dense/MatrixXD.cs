using EigenCore.Core.Dense.Complex;
using EigenCore.Core.Dense.LinearAlgebra;
using EigenCore.Eigen;
using System.Linq;

namespace EigenCore.Core.Dense
{
    public class MatrixXD : MatrixDenseBase<double>
    {
        private static double[] JaggedToFlatColumnWise(double[][] inputValues)
        {
            var numberOfRows = inputValues.Length;
            var numberOfCols = inputValues[0].Length;
            double[] values = new double[numberOfRows * numberOfCols];
            int index = 0;
            for (int col = 0; col < numberOfCols; col++)
            {
                for (int row = 0; row < numberOfRows; row++)
                {
                    values[index] = inputValues[row][col];
                    index += 1;
                }
            }

            return values;
        }

        private static double[] MultyDimToFlatColumnWise(double[,] inputValues)
        {
            var numberOfRows = inputValues.GetLength(0);
            var numberOfCols = inputValues.GetLength(1);
            double[] values = new double[numberOfRows * numberOfCols];
            int index = 0;
            for (int col = 0; col < numberOfCols; col++)
            {
                for (int row = 0; row < numberOfRows; row++)
                {
                    values[index] = inputValues[row,col];
                    index += 1;
                }
            }

            return values;
        }

        private static (int, int) JaggedRowsAndColsInfo(double[][] inputValues) => (inputValues.Length, inputValues[0].Length);

        private static (int, int) MultDimRowsAndColsInfo(double[,] inputValues) => (inputValues.GetLength(0), inputValues.GetLength(1));

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

        public double Max() => _values.Max();

        public double Min() => _values.Min();

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
