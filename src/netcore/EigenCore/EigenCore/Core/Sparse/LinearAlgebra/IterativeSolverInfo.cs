namespace EigenCore.Core.Sparse.LinearAlgebra
{
    public class IterativeSolverInfo
    {
        public IterativeSolverType Solver { get; }
        public int MaxIterations { get; }
        public double Tolerance { get; }

        public IterativeSolverInfo(IterativeSolverType solver = IterativeSolverType.ConjugateGradient, int maxIterations = -1, double tolerance = -1)
        {
            Solver = solver;
            MaxIterations = maxIterations;
            Tolerance = tolerance;
        }
    }
}
