namespace EigenCore.Core.Dense.LinearAlgebra
{
    public class QRResult
    {
        public MatrixXD Q { get; }
        public MatrixXD R { get; }
        public MatrixXD P { get; }

        public QRResult(MatrixXD q, MatrixXD r, MatrixXD p = default(MatrixXD))
        {
            Q = q;
            R = r;
            P = p;
        }
    }
}
