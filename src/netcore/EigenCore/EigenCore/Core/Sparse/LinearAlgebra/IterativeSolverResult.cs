using EigenCore.Core.Dense;

namespace EigenCore.Core.Sparse.LinearAlgebra
{
    public class IterativeSolverResult
    {
        public IterativeSolverResult(VectorXD result, int interations, double error, IterativeSolverType solver)
        {
            Result = result;
            Interations = interations;
            Error = error;
            Solver = solver;
        }

        public VectorXD Result { get; }
        public int Interations { get; }
        public double Error { get; }
        public IterativeSolverType Solver { get; }
    }
}
