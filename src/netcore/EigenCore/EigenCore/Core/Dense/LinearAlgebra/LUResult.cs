namespace EigenCore.Core.Dense.LinearAlgebra
{
    public class LUResult
    {
        public MatrixXD L { get; }
        public MatrixXD U { get; }

        public LUResult(MatrixXD L, MatrixXD U)
        {
            this.L = L;
            this.U = U;
        }
    }
}
