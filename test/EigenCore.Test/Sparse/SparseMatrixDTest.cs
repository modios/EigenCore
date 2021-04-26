using EigenCore.Core.Dense;
using EigenCore.Core.Sparse;
using EigenCore.Core.Sparse.LinearAlgebra;
using Xunit;

namespace EigenCore.Test.Sparse
{
    public class SparseMatrixDTest
    {
        public const int DoublePrecision = 12;
        
        [Fact]
        public void ConstructorListTuples_ShouldSucced()
        {
            (int, int, double)[] elements = {
                (0, 1, 3.0),
                (1, 0, 22),
                (2, 0, 7),
                (2, 1, 5),
                (4, 2, 14),
                (2, 3, 1),
                (1, 4, 17),
                (4,4,8)
            };

            SparseMatrixD A = new SparseMatrixD(elements, 5, 5);

            foreach (var element in elements)
            {
                Assert.Equal(element.Item3, A.Get(element.Item1, element.Item2));
            }
        }

        [Fact(Skip = "need to update .so")]
        public void ADD_ShouldSucced()
        {
            (int, int, double)[] elements = {
                (0, 0, 6),
                (0, 1, 4),
                (0, 2, 0),
                (1, 0, 4),
                (1 ,1, 4),
                (1, 2, 1),
                (2, 0, 0),
                (2, 1, 1),
                (2, 2, 8)
            };

            SparseMatrixD A = new SparseMatrixD(elements, 3, 3);
            var result = A.Add(SparseMatrixD.Identity(3));

            Assert.Equal(new MatrixXD("7 4 0;4 5 1; 0 1 9"), result.ToDense());
        }

        [Fact(Skip = "need to update .so")]
        public void Minus_ShouldSucced()
        {
            (int, int, double)[] elements = {
                (0, 0, 6),
                (0, 1, 4),
                (0, 2, 0),
                (1, 0, 4),
                (1 ,1, 4),
                (1, 2, 1),
                (2, 0, 0),
                (2, 1, 1),
                (2, 2, 8)
            };

            SparseMatrixD A = new SparseMatrixD(elements, 3, 3);
            var result = A.Minus(SparseMatrixD.Identity(3));

            Assert.Equal(new MatrixXD("5 4 0;4 3 1; 0 1 7"), result.ToDense());


            result = A.Minus(A);
            Assert.Equal(MatrixXD.Zeros(3,3), result.ToDense());
        }

        [Fact(Skip = "need to update .so")]
        public void Mult_ShouldSucced()
        {
            (int, int, double)[] elements = {
                (0, 0, 6),
                (0, 1, 4),
                (1, 0, 4),
                (1 ,1, 4),
                (1, 2, 1),
                (2, 1, 1),
                (2, 2, 8)
            };

            SparseMatrixD A = new SparseMatrixD(elements, 3, 3);
            var result = A.Mult(A);

            Assert.Equal(new MatrixXD("52 40 4;40 33 12;4 12 65"), result.ToDense());


            SparseMatrixD B = new MatrixXD("0 0 0;1.4 0 0;0 1.9 0;0 0 0").ToSparse();
            SparseMatrixD C = new MatrixXD("3 0 0;0 1.2 0;0 3.2 0;0 0 0").ToSparse();
            result = B.Mult(C);
            Assert.Equal(new MatrixXD("0 0 0;4.199999999999999 0 0;0 2.28 0;0 0 0"), result.ToDense());
        }

        [Fact(Skip = "need to update .so")]
        public void MultWithVector_ShouldSucced()
        {
            SparseMatrixD A = MatrixXD.Diag(new[] { 3.5, 2, 4.5 }).ToSparse();
            VectorXD v = new VectorXD(new[] { 2.0, 2.0, 2.0 });
            var result = A.Mult(v);
            Assert.Equal(new double[] { 7, 4, 9 }, result.GetValues().ToArray());


            A = new MatrixXD("1 2 1; 2 5 2; 2 5 5").ToSparse();
            v = new VectorXD(new[] { 1.5, 4.5, 2.0 });
            result = A.Mult(v);
            Assert.Equal(new double[] { 12.5, 29.5, 35.5 }, result.GetValues().ToArray());

            A = new MatrixXD("1 -2 ; 2 5 ; 4 -2").ToSparse();
            v = new VectorXD(new[] { 3.0, -7.0 });
            result = A.Mult(v);
            Assert.Equal(new double[] { 17.0, -29.0, 26.0 }, result.GetValues().ToArray());
        }

        [Fact(Skip = "need to update .so")]
        public void ConjugateGradient_ShouldSucced()
        {
            (int, int, double)[] elements = {
                (0, 0, 6),
                (0, 1, 4),
                (0, 2, 0),
                (1, 0, 4),
                (1 ,1, 4),
                (1, 2, 1),
                (2, 0, 0),
                (2, 1, 1),
                (2, 2, 8)
            };

            SparseMatrixD A = new SparseMatrixD(elements, 3, 3);
            var rhs = new VectorXD("3 3 4");
            var result = A.Solve(rhs);
            Assert.Equal(new VectorXD("0.22413793103448287 0.41379310344827569 0.44827586206896558"), result.Result);
            Assert.Equal(2, result.Interations);
            Assert.Equal(0, result.Error, DoublePrecision);

            result = A.Solve(rhs, new IterativeSolverInfo(IterativeSolverType.ConjugateGradient, 2, 1e-2));
            Assert.Equal(new VectorXD("0.22758069267348396 0.4094769178975719 0.44892479905735805"), result.Result);
            Assert.Equal(1, result.Interations);
            Assert.Equal(0.00077389987808970792, result.Error, DoublePrecision);
        }

        [Fact(Skip = "need to update .so")]
        public void BiCGSTAB_ShouldSucced()
        {
            (int, int, double)[] elements = {
                (0, 0, 6),
                (0, 1, 4),
                (0, 2, 0),
                (1, 0, 4),
                (1 ,1, 4),
                (1, 2, 1),
                (2, 0, 0),
                (2, 1, 1),
                (2, 2, 8)
            };

            SparseMatrixD A = new SparseMatrixD(elements, 3, 3);
            var rhs = new VectorXD("3 3 4");
            var result = A.Solve(rhs, new IterativeSolverInfo(IterativeSolverType.BiCGSTAB));
            Assert.Equal(new VectorXD("0.22413793103448287 0.41379310344827569 0.44827586206896558"), result.Result);
            Assert.Equal(3, result.Interations);
            Assert.Equal(0, result.Error, DoublePrecision);

            result = A.Solve(rhs, new IterativeSolverInfo(IterativeSolverType.BiCGSTAB, 2, 1e-2));
            Assert.Equal(new VectorXD("0.22615806890721962 0.41138770777252714 0.44877544841531969"), result.Result);
            Assert.Equal(2, result.Interations);
            Assert.Equal(0.00053859978391084472, result.Error, DoublePrecision);
        }
    }
}
