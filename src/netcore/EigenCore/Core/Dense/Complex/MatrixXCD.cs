using System.Linq;

namespace EigenCore.Core.Dense
{
    public class MatrixXCD : VBufferDenseComplex<double>
    {
        public int Rows { get; }

        public int Cols { get; }

        public MatrixXD Real()
        {
            return new MatrixXD(_realValues.ToArray(), Rows, Cols);
        }

        public MatrixXD Imag()
        {
            return new MatrixXD(_imagValues.ToArray(), Rows, Cols);
        }

        public MatrixXCD(double[] realValues, double[] complexValues, int rows, int cols)
            : base(realValues, complexValues)
        {
            Rows = rows;
            Cols = cols;
        }
    }
}
