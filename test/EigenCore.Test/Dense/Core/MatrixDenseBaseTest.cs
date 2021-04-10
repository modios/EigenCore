using EigenCore.Core.Dense;
using Xunit;

namespace EigenCore.Test.Dense.Core
{
    public sealed class MatrixDenseBaseTest
    {
        [Fact]
        public void ToString_ShouldSucceed()
        {
            var A = new MatrixXD(new double[][] { new double[] { 1, 3, 1 }, new double[] { 0, 2, 1 } });
            Assert.Equal("MatrixXD, 2 * 3:\n\n1.00 3.00 1.00 \n0.00 2.00 1.00 \n\n", A.ToString());
        }

        [Fact]
        public void ConstructorString_ShouldSucceed()
        {
            double[,] values = new double[,] {
                { 1.0, 2.0 },
                { 3.0, 5.0 },
                { 7.0, 9.0 }
            };

            int rows = 3;
            int cols = 2;
            MatrixXD A = new MatrixXD("1 2; 3 5; 7 9", 3, 2);
            Assert.Equal(3, A.Rows);
            Assert.Equal(2, A.Cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    Assert.Equal(values[i, j], A.Get(i, j));
                }
            }
        }
    }
}
