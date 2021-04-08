namespace EigenCore.Core.Dense
{
    public class MatrixXCD : VBufferDenseComplex<double>
    {
        public MatrixXCD(double[] realValues, double[] complexValues)
            : base(realValues, complexValues)
        {
        }
    }
}
