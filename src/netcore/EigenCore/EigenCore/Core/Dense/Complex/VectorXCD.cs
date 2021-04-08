using System.Linq;

namespace EigenCore.Core.Dense.Complex
{
    public class VectorXCD : VBufferDenseComplex<double>
    {

        public VectorXD Real()
        {
            return new VectorXD(_realValues.ToArray());
        }

        public VectorXD Imag()
        {
            return new VectorXD(_imagValues.ToArray());
        }

        public VectorXCD(double[] realValues, double[] imagValues) 
            : base(realValues, imagValues)
        {
        }
    }
}
