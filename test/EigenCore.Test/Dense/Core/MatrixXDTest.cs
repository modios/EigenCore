using EigenCore.Core.Dense;
using Xunit;

namespace EigenCore.Test.Dense.Core
{
    public class MatrixXDTest
    {
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
       
            Assert.Equal(new double[] { 11 ,35, 0, 20}, result.GetValues().ToArray());
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
            MatrixXD A = MatrixXD.Identity(3, 3);
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
    }
}
