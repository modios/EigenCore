using EigenCore.Core.Dense;
using System;
using Xunit;

namespace EigenCore.Test.Dense.Core
{
    public class MatrixXDTest
    {
        [Fact]
        public void ConstructorJagged_ShouldSucceed()
        {
            var A = new MatrixXD(new double[][] { new double[] { 1, 3, 2 } , new double[] { 0, 2, 1 } });
            Assert.Equal(2, A.Rows);
            Assert.Equal(3, A.Cols);
            Assert.Equal(new double[] { 1, 0, 3, 2, 2, 1 }, A.GetValues().ToArray());
        }

        [Fact]
        public void ConstructorMutyDim_ShouldSucceed()
        {
            var A = new MatrixXD(new double[,] { { 1, 3, 2 }, { 0, 2, 1 } });
            Assert.Equal(2, A.Rows);
            Assert.Equal(3, A.Cols);
            Assert.Equal(new double[] { 1, 0, 3, 2, 2, 1 }, A.GetValues().ToArray());
        }

        [Fact]
        public void Mult_ShouldSucceed()
        {

            MatrixXD A = new MatrixXD("1 2; 3 5", 2, 2);
            MatrixXD B = new MatrixXD("1 2; 3 2", 2, 2);

            var result = A.Mult(B);
            Assert.Equal(new double[] { 7, 18, 6, 16 }, result.GetValues().ToArray());

            A = new MatrixXD("0 -1 2; 4 11 2", 2, 3);
            B = new MatrixXD("3, -1; 1 2; 6 1", 3, 2);

            result = A.Mult(B);

            Assert.Equal(new double[] { 11, 35, 0, 20 }, result.GetValues().ToArray());
        }

        [Fact(Skip = "need to update .so")]
        public void Transpose_ShouldSucceed()
        {
            MatrixXD A = new MatrixXD("1 2 4; 3 5 7", 2, 3);
            MatrixXD B = A.Transpose();
            Assert.Equal(new double[] { 1, 2, 4, 3, 5, 7 }, B.GetValues().ToArray());
        }


        [Fact(Skip = "need to update .so")]
        public void MultT_ShouldSucceed()
        {
            MatrixXD A = new MatrixXD("1 2 1; 2 5 2", 2, 3);
            MatrixXD B = new MatrixXD("1 0 1; 1 1 0", 2, 3);
            MatrixXD C = A.MultT(B);
            Assert.Equal(new double[] { 2, 4, 3, 7 }, C.GetValues().ToArray());
        }

        [Fact]
        public void Random_ShouldSucceed()
        {
            MatrixXD A = MatrixXD.Random(10, 10, 1, 100);
            Assert.Equal(10, A.Rows);
            Assert.Equal(10, A.Cols);
            Assert.True(A.Min() >= 1);
            Assert.True(A.Max() <= 100);
        }

        [Fact]
        public void Identity_ShouldSucceed()
        {
            MatrixXD A = MatrixXD.Identity(3);
            Assert.Equal(3, A.Rows);
            Assert.Equal(3, A.Cols);
            Assert.True(A.Min() == 0.0);
            Assert.True(A.Max() == 1.0);
            Assert.Equal(
                new double[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 },
                A.GetValues().ToArray());
        }

        [Fact]
        public void Equals_ShouldSucceed()
        {
            MatrixXD A = MatrixXD.Random(10, 10);
            Assert.Equal(10, A.Rows);
            Assert.Equal(10, A.Cols);
            Assert.True(A.Equals(A.Clone()));
        }

        [Fact]
        public void Clone_ShouldSucceed()
        {
            MatrixXD A = MatrixXD.Random(10, 10);
            Assert.Equal(10, A.Rows);
            Assert.Equal(10, A.Cols);
            Assert.True(A.Min() >= 0.0);
            Assert.True(A.Max() <= 1.0);
        }

        [Fact]
        public void Zeros_ShouldSucceed()
        {
            MatrixXD A = MatrixXD.Zeros(2, 3);
            Assert.Equal(2, A.Rows);
            Assert.Equal(3, A.Cols);
            Assert.Equal(0.0, A.Min());
            Assert.Equal(0.0, A.Max());

            Assert.Equal(new double[] { 0, 0, 0, 0, 0, 0 }, A.GetValues().ToArray());
        }

        [Fact]
        public void Ones_ShouldSucceed()
        {
            MatrixXD A = MatrixXD.Ones(2, 3);
            Assert.Equal(2, A.Rows);
            Assert.Equal(3, A.Cols);
            Assert.Equal(1.0, A.Min());
            Assert.Equal(1.0, A.Max());

            Assert.Equal(new double[] { 1, 1, 1, 1, 1, 1 }, A.GetValues().ToArray());
        }

        [Fact]
        public void Diag_ShouldSucceed()
        {
            MatrixXD A = MatrixXD.Diag(new[] { 3.5, 2, 4.5 });
            Assert.Equal(3, A.Rows);
            Assert.Equal(3, A.Cols);
            Assert.Equal(0.0, A.Min());
            Assert.Equal(4.5, A.Max());

            Assert.Equal(new double[] { 3.5, 0, 0, 0, 2, 0, 0, 0, 4.5 }, A.GetValues().ToArray());
        }


        [Fact(Skip = "need to update .so")]
        public void MultV_ShouldSucceed()
        {
            MatrixXD A = MatrixXD.Diag(new[] { 3.5, 2, 4.5 });
            VectorXD v = new VectorXD(new[] { 2.0, 2.0, 2.0 });
            var result = A.Mult(v);
            Assert.Equal(new double[] { 7, 4, 9 }, result.GetValues().ToArray());
        }

        [Fact(Skip = "need to update .so")]
        public void Trace_ShouldSucceed()
        {
            MatrixXD A = MatrixXD.Diag(new[] { 3.5, 2, 4.5 });
            var result = A.Trace();
            Assert.Equal(10.0, result);
        }

        [Fact(Skip = "need to update .so")]
        public void Eigen_ShouldSucceed()
        {
            MatrixXD A = MatrixXD.Diag(new[] { 3.5, 2, 4.5 });
            var result = A.Eigen();
            Assert.Equal(new[] { 3.5, 2, 4.5 }, result.Item1.Real().GetValues().ToArray());
        }

        [Fact(Skip = "need to update .so")]
        public void PlusT_ShouldSucceed()
        {
            MatrixXD A = new MatrixXD("2 2; 1 1", 2, 2);
            var result = A.PlusT();
            Assert.Equal(new double[] { 4, 3, 3 , 2 }, result.GetValues().ToArray());
        }

        [Fact(Skip = "need to update .so")]
        public void SymmetricEigen_ShouldSucceed()
        {
            MatrixXD A = new MatrixXD("4 3; 3 2", 2, 2);
            var eigen = A.SymmetricEigen();
            Assert.Equal(new[] { -0.16227766016837947, 6.162277660168379 }, eigen.Item1.GetValues().ToArray());
        }

        [Fact(Skip = "need to update .so")]
        public void Plus_ShouldSucceed()
        {
            MatrixXD A = new MatrixXD("4 3; 3 2", 2, 2);
            MatrixXD B = MatrixXD.Identity(2);
            var result = A.Plus(B);
            Assert.Equal(new double[] { 5, 3, 3, 3 }, result.GetValues().ToArray());
        }
    }
}
