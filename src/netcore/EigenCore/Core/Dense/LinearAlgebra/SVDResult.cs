namespace EigenCore.Core.Dense.LinearAlgebra
{
    public class SVDResult
    {
        public MatrixXD U { get; }

        public VectorXD S { get; }

        public MatrixXD V { get; }

        public SVDResult(MatrixXD u, VectorXD s, MatrixXD v)
        {
            U = u;
            S = s;
            V = v;
        }
    }
}
