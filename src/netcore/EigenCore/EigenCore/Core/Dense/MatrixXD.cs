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

        public double Max() => _values.Max();

        public double Min() => _values.Min();

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

        public static MatrixXD Identity(int rows, int cols)
        {
            double[] input = new double[rows * cols];

            for (int i = 0; i < cols; i++)
            {  
                input[i * (cols + 1)] = 1.0;
            }

            return new MatrixXD(input, rows, cols);
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

        public MatrixXD(double[] values, int rows, int cols)
                : base(values.ToArray(), rows, cols)
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
