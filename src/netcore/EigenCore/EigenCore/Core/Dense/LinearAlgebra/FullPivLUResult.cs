namespace EigenCore.Core.Dense.LinearAlgebra
{
    /// <summary>
    /// A = P^-1LUQ^-1
    /// </summary>
    public class FullPivLUResult : LUResult
    {
        public MatrixXD P { get; }
        public MatrixXD Q { get; }

        public FullPivLUResult(MatrixXD l, MatrixXD u, MatrixXD p, MatrixXD q) : base(l, u)
        {
            P = p;
            Q = q;
        }
    } 
}
