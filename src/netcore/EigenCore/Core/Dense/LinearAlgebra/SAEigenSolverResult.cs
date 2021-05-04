namespace EigenCore.Core.Dense.LinearAlgebra
{
    public class SAEigenSolverResult
    {
        public VectorXD Eigenvalues { get; }
        public MatrixXD Eigenvectors { get; }

        public SAEigenSolverResult(VectorXD eigenvalues, MatrixXD eigenvectors)
        {
            Eigenvalues = eigenvalues;
            Eigenvectors = eigenvectors;
        }
    }
}
