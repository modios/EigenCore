namespace EigenCore.Core.Dense.Complex
{
    public class VectorXDComplex : VBufferDenseComplex<double>
    {
        public VectorXDComplex(double[] realValues, double[] complexValues) 
            : base(realValues, complexValues)
        {
        }
    }
}
