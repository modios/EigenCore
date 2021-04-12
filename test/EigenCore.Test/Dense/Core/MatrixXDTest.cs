using EigenCore.Core.Dense;
using EigenCore.Core.Dense.LinearAlgebra;
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

        [Fact(Skip = "need to update .so")]
        public void Minus_ShouldSucceed()
        {

            MatrixXD A = new MatrixXD("2 4; 3 5");
            MatrixXD B = new MatrixXD("1 3; 3 2");

            var result = A.Minus(B);
            Assert.Equal(new MatrixXD("1 1; 0 3"), result);
        }

        [Fact]
        public void Mult_ShouldSucceed()
        {

            MatrixXD A = new MatrixXD("1 2; 3 5");
            MatrixXD B = new MatrixXD("1 2; 3 2");

            var result = A.Mult(B);
            Assert.Equal(new double[] { 7, 18, 6, 16 }, result.GetValues().ToArray());

            A = new MatrixXD("0 -1 2; 4 11 2");
            B = new MatrixXD("3, -1; 1 2; 6 1");

            result = A.Mult(B);

            Assert.Equal(new double[] { 11, 35, 0, 20 }, result.GetValues().ToArray());
        }

        [Fact(Skip = "need to update .so")]
        public void Transpose_ShouldSucceed()
        {
            MatrixXD A = new MatrixXD("1 2 4; 3 5 7");
            MatrixXD B = A.Transpose();
            Assert.Equal(new double[] { 1, 2, 4, 3, 5, 7 }, B.GetValues().ToArray());
        }

        [Fact(Skip = "need to update .so")]
        public void MultT_ShouldSucceed()
        {
            MatrixXD A = new MatrixXD("1 2 1; 2 5 2");
            MatrixXD B = new MatrixXD("1 0 1; 1 1 0");
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


            A = new MatrixXD("1 2 1; 2 5 2; 2 5 5");
            v = new VectorXD(new[] { 1.5, 4.5, 2.0 });
            result = A.Mult(v);
            Assert.Equal(new double[] { 12.5, 29.5, 35.5 }, result.GetValues().ToArray());

            A = new MatrixXD("1 -2 ; 2 5 ; 4 -2");
            v = new VectorXD(new[] { 3.0, -7.0 });
            result = A.Mult(v);
            Assert.Equal(new double[] { 17.0, -29.0, 26.0 }, result.GetValues().ToArray());
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

            var v1 = result.Eigenvalues.Real();
            var v2 = result.Eigenvectors.Real();

            Assert.Equal(new[] { 3.5, 2, 4.5 }, v1.GetValues().ToArray());
            Assert.Equal(new[] { 1.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0,
                                 0.0, 0.0, 1.0 }, v2.GetValues().ToArray());

            MatrixXD B = new MatrixXD("0 1; -2 -3");
            result = B.Eigen();
            v1 = result.Eigenvalues.Real();
            v2 = result.Eigenvectors.Real();

            Assert.Equal(new[] { -1.0, -2.0 }, v1.GetValues().ToArray());
            Assert.Equal(new[] { 0.7071067811865476, -0.7071067811865475,
                                -0.447213595499958, 0.8944271909999157 }, v2.GetValues().ToArray());
        }

        [Fact(Skip = "need to update .so")]
        public void PlusT_ShouldSucceed()
        {
            MatrixXD A = new MatrixXD("2 2; 1 1");
            var result = A.PlusT();
            Assert.Equal(new double[] { 4, 3, 3 , 2 }, result.GetValues().ToArray());
        }

        [Fact(Skip = "need to update .so")]
        public void MultT_WithSelf_ShouldSucceed()
        {
            MatrixXD A = new MatrixXD("2 2; 1 1");
            var result = A.MultT();
            Assert.Equal(new MatrixXD("8 4 ; 4 2"), result);

            MatrixXD B = new MatrixXD("2 2; 1 1; 3 3");
            result = B.MultT();
            Assert.Equal(new MatrixXD("8 4 12; 4 2 6; 12 6 18"), result);
        }

        [Fact(Skip = "need to update .so")]
        public void TMult_WithSelf_ShouldSucceed()
        {
            MatrixXD A = new MatrixXD("2 2; 1 1");
            var result = A.TMult();
            Assert.Equal(new MatrixXD("5 5 ; 5 5"), result);

            MatrixXD B = new MatrixXD("2 2; 1 1; 3 3");
            result = B.TMult();
            Assert.Equal(new MatrixXD("14 14; 14 14"), result);
        }

        [Fact(Skip = "need to update .so")]
        public void SymmetricEigen_ShouldSucceed()
        {
            MatrixXD A = new MatrixXD("4 3; 3 2");
            SAEigenSolverResult eigen = A.SymmetricEigen();
            Assert.Equal(new[] { -0.16227766016837947, 6.162277660168379 }, eigen.Eigenvalues.GetValues().ToArray());
            Assert.Equal(new[] { -0.584710284663765, 0.8112421851755608,
                                  0.8112421851755608, 0.584710284663765 }, eigen.Eigenvectors.GetValues().ToArray());

            MatrixXD B = new MatrixXD("2 1; 1 2");
            eigen = B.SymmetricEigen();
            Assert.Equal(new[] { 0.9999999999999998, 2.9999999999999996 }, eigen.Eigenvalues.GetValues().ToArray());
            Assert.Equal(new[] { -0.7071067811865475, 0.7071067811865475,
                                  0.7071067811865475, 0.7071067811865475 }, eigen.Eigenvectors.GetValues().ToArray());
        }

        [Fact(Skip = "need to update .so")]
        public void Plus_ShouldSucceed()
        {
            MatrixXD A = new MatrixXD("4 3; 3 2");
            MatrixXD B = MatrixXD.Identity(2);
            var result = A.Plus(B);
            Assert.Equal(new MatrixXD("5 3; 3 3"), result);
        }

        [Theory(Skip = "need to update .so")]
        [InlineData(SVDType.Jacobi)]
        [InlineData(SVDType.BdcSvd)]
        public void SVD_ShouldSucceed(SVDType sdvType)
        {
            var A = new MatrixXD("3 2 2 ; 2 3 -2");
            SVDResult result = A.SVD(sdvType);

            Assert.Equal(new MatrixXD("-0.7071067811865476 0.7071067811865475 ;" +
                " -0.7071067811865475  -0.7071067811865476"), result.U);

            Assert.Equal(new[] { 5.0, 3.0 }, result.S.GetValues().ToArray());

            Assert.Equal(new MatrixXD("-0.7071067811865477  0.23570226039551567; " +
                "-0.7071067811865475 -0.23570226039551567; " +
                "-2.220446049250313E-16 0.94280904158206336"),
                result.V);
        }

        [Theory(Skip = "need to update .so")]
        [InlineData(SVDType.Jacobi)]
        [InlineData(SVDType.BdcSvd)]
        public void LeastSquaresSVD_ShouldSucceed(SVDType sdvType)
        {
            var A = new MatrixXD("-1 -0.0827; -0.737 0.0655; 0.511 -0.562 ");
            var rhs = new VectorXD("-0.906 0.358 0.359");
            VectorXD result = A.LeastSquaresSVD(rhs, sdvType);
            Assert.Equal(new VectorXD("0.46347421844577846 0.04209165616389611"), result);
        }
    }
}
