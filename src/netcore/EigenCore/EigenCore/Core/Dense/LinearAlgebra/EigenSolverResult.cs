using EigenCore.Core.Dense.Complex;

namespace EigenCore.Core.Dense.LinearAlgebra
{
    public class EigenSolverResult
    {
        public VectorXCD Eigenvalues { get; }
        public MatrixXCD Eigenvectors { get; }

        public EigenSolverResult(VectorXCD eigenvalues, MatrixXCD eigenvectors)
        {
            Eigenvalues = eigenvalues;
            Eigenvectors = eigenvectors;
        }
    }
}
