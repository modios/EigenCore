using EigenCore.Eigen;
using System;
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

        public double Max() => _values.Max();

        public double Min() => _values.Min();

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
            var length = other.Length;
            double[] outVector = new double[length];
            EigenDenseUtilities.Mult(GetValues(),
                Rows,
                Cols,
                other.GetValues(),
                other.Length,
                outVector);
            return new VectorXD(outVector);
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
            EigenDenseUtilities.MultT(GetValues(), Rows, Cols, other.GetValues(), other.Rows, other.Cols,  outMatrix);
            return new MatrixXD(outMatrix, Rows, other.Rows);
        }

        public double Trace()
        {
            return EigenDenseUtilities.Trace(GetValues(), Rows, Cols);
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

        protected MatrixXD(double[] values, int rows, int cols)
                : base(values, rows, cols)
        {
        }

        public MatrixXD(MatrixXD matrixXD)
        : base(matrixXD._values.ToArray(), matrixXD.Rows, matrixXD.Cols)
        {
        }

        public MatrixXD(string valuesString, int rows, int cols)
            : base(valuesString, rows, cols, (string value) => double.Parse(value))
        {

        }
    }
}
