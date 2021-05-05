using EigenCore.Core.Dense;
using EigenCore.Core.Sparse;
using Xunit;

namespace EigenCore.Test.Core.Sparse
{
    public sealed class MatrixSparseBaseTest
    {
        [Fact]
        public void Count_ShouldSucceed()
        {
            var A = new MatrixXD(new double[][] { new double[] { 1, 3, -1}, new double[] { 0, double.PositiveInfinity, -1 } }).ToSparse();

            Assert.Equal(2, A.Count(x => x == -1));
            Assert.Equal(1, A.Count(x => x == 0));
            Assert.Equal(1, A.Count(x => double.IsInfinity(x)));
        }

        [Fact]
        public void Replace_ShouldSucceed()
        {
            var A = new MatrixXD(new double[][] {
                new double[] { 1, 3, -1},
                new double[] { 0, double.PositiveInfinity, -1 }
            }).ToSparse();

            A.Replace(x => x == -1 ? 0.0 : x);
            Assert.Equal(0, A.Get(0, 2));
            Assert.Equal(0, A.Get(1, 2));
            A.Replace(x => double.IsInfinity(x) ? 0.0 : x);
            Assert.Equal(0, A.Get(1, 1));
        }

        [Fact]
        public void ToString_ShouldSucceed()
        {
            var A = new MatrixXD(new double[][] { new double[] { 1, 3, 1 }, new double[] { 0, 2, 1 } }).ToSparse();
            Assert.Equal("SparseMatrixD, 2 * 3:\n\n1 3 1 \n0 2 1 \n\n", A.ToString());

            var B = new MatrixXD(new double[][] {
                new double[] { 1.3423432, 3.234324, 3243241 },
                new double[] { 0.32432, 3243242, 0.32432431 } }).ToSparse();
            Assert.Equal("SparseMatrixD, 2 * 3:\n\n1.34 3.23 3.24E+06 \n0.324 3.24E+06 0.324 \n\n", B.ToString());
        }
    }
}
