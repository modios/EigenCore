namespace EigenCore.Core.Dense.LinearAlgebra
{
    public class QRResult
    {
        public MatrixXD Q { get; }
        public MatrixXD R { get; }

        public QRResult(MatrixXD q, MatrixXD r)
        {
            Q = q;
            R = r;
        }
    }
}
