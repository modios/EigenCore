using EigenCore.Eigen;

namespace EigenCore.Core.Dense
{
    public class VectorXD : VBufferDense<double>
    {
        public double Dot(VectorXD other)
        {
            return EigenDenseUtilities.Dot(GetValues(), other.GetValues(), Length);
        }

        public VectorXD Add(VectorXD other)
        {
            double[] outVector = new double[Length];
            EigenDenseUtilities.Add(GetValues(), other.GetValues(), Length, outVector);
            return new VectorXD(outVector);
        }

        public VectorXD Scale(double scalar)
        {
            double[] outVector = new double[Length];
            EigenDenseUtilities.Scale(GetValues(), scalar, Length, outVector);
            return new VectorXD(outVector);
        }

        public VectorXD(double[] values) : base(values)
        {
        }
    }
}
