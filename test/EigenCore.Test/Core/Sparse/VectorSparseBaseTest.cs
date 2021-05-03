using EigenCore.Core.Dense;
using EigenCore.Core.Sparse;
using Xunit;

namespace EigenCore.Test.Core.Sparse
{
    public class VectorSparseBaseTest
    {
        [Fact]
        public void Count_ShouldSucceed()
        {
            var A = new VectorXD(new double[] { 1, 3, -1, 0, double.PositiveInfinity, -1 }).ToSparse();

            Assert.Equal(1, A.Count(x => x == 0));
            Assert.Equal(2, A.Count(x => x == -1));
            Assert.Equal(1, A.Count(x => double.IsInfinity(x)));
        }

        [Fact]
        public void Replace_ShouldSucceed()
        {
            var A = new VectorXD(new double[] { 1, 3, -1, 0, double.PositiveInfinity, -1 }).ToSparse();

            A.Replace(x => x == -1 ? 0.0 : x);
            Assert.Equal(0, A.Get(2));
            Assert.Equal(0, A.Get(5));
            A.Replace(x => double.IsInfinity(x) ? 0.0 : x);
            Assert.Equal(0, A.Get(4));
        }

        [Fact]
        public void ToString_ShouldSucceed()
        {
            SparseVectorD A = new VectorXD(new double[] { 1, 3, 1 }).ToSparse();
            Assert.Equal("SparseVectorD, 3:\n\n1 3 1", A.ToString());

            var B = new VectorXD(new double[] { 1.3423432, 3.234324, 3243241 }).ToSparse();
            Assert.Equal("SparseVectorD, 3:\n\n1.34 3.23 3.24E+06", B.ToString());

            var C = VectorXD.Linespace(0, 99, 100).ToSparse();
            Assert.Equal("SparseVectorD, 100:\n\n0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ...", C.ToString());
        }
    }
}
