using EigenCore.Core.Dense;
using Xunit;

namespace EigenCore.Test.Dense.Core
{
    public sealed class MatrixDenseBaseTest
    {
        [Fact]
        public void Count_ShouldSucceed()
        {
            var A = new MatrixXD(new double[][] { new double[] { 1, 3, double.NaN }, new double[] { 0, double.PositiveInfinity, double.NaN } });

            Assert.Equal(2, A.Count(x => double.IsNaN(x)));
            Assert.Equal(1, A.Count(x => double.IsInfinity(x)));       
        }

        [Fact]
        public void Replace_ShouldSucceed()
        {
            var A = new MatrixXD(new double[][] { new double[] { 1, 3, double.NaN }, new double[] { 0, double.PositiveInfinity, double.NaN } });

            A.Replace(x => double.IsNaN(x) ? 0.0 : x);
            Assert.Equal(0, A.Get(0, 2));
            Assert.Equal(0, A.Get(1, 2));
            A.Replace(x => double.IsInfinity(x) ? 0.0 : x);
            Assert.Equal(0, A.Get(1, 1));
        }

        [Fact]
        public void ToString_ShouldSucceed()
        {
            var A = new MatrixXD(new double[][] { new double[] { 1, 3, 1 }, new double[] { 0, 2, 1 } });
            Assert.Equal("MatrixXD, 2 * 3:\n\n1 3 1 \n0 2 1 \n\n", A.ToString());

            var B = new MatrixXD(new double[][] {
                new double[] { 1.3423432, 3.234324, 3243241 },
                new double[] { 0.32432, 3243242, 0.32432431 } });
            Assert.Equal("MatrixXD, 2 * 3:\n\n1.34 3.23 3.24E+06 \n0.324 3.24E+06 0.324 \n\n", B.ToString());
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
            MatrixXD A = new MatrixXD("1 2; 3 5; 7 9");
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
