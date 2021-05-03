using EigenCore.Core.Dense;
using EigenCore.Core.Sparse;
using System.Linq;
using Xunit;

namespace EigenCore.Test.Core.Sparse
{
    public class SparseVectorDTest
    {
        [Fact]
        public void Random_ShouldSucceed()
        {
            var A = SparseVectorD.Random(100, 0.1, 2, 200);
            Assert.True(A.Max() <= 200);
            Assert.True(A.Min() == 0);
            Assert.Equal(10, A.Nnz);
            Assert.Equal(100, A.Length);
        }

        [InlineData("2 2 1", 3)]
        [InlineData("1 0 0 0", 1)]
        [Theory]
        public void Norm_ShouldSucceed(string valuesString, double expected)
        {
            var v = new VectorXD(valuesString).ToSparse();
            Assert.Equal(expected, v.Norm());
        }

        [InlineData("2 2 1", 9)]
        [InlineData("1 0 0 0", 1)]
        [Theory]
        public void SquaredNorm_ShouldSucceed(string valuesString, double expected)
        {
            var v = new VectorXD(valuesString).ToSparse();
            Assert.Equal(expected, v.SquaredNorm());
        }

        [InlineData("2 2 1", 5)]
        [InlineData("2 2 -1", 5)]
        [InlineData("1 0 0 0", 1)]
        [Theory]
        public void Lp1Norm_ShouldSucceed(string valuesString, double expected)
        {
            var v = new VectorXD(valuesString).ToSparse();
            Assert.Equal(expected, v.Lp1Norm());
        }

        [InlineData("2 2 1", 2)]
        [InlineData("2 2 -3", 3)]
        [InlineData("1 0 0 0", 1)]
        [Theory]
        public void LpInfNorm_ShouldSucceed(string valuesString, double expected)
        {
            var v = new VectorXD(valuesString).ToSparse();
            Assert.Equal(expected, v.LpInfNorm());
        }

        [Fact]
        public void Constructor_ShouldSucced()
        {
            var elements = new[] { (0, 1.0), (5, 2.0), (3, 9.0), (19, 20) };
            SparseVectorD vector = new SparseVectorD(elements, 20);

            Assert.Equal(new[] { 0, 3, 5, 19 }, vector.GetIndices().ToArray());
            Assert.Equal(new[] { 1.0, 9.0, 2.0, 20 }, vector.GetValues().ToArray());
            Assert.Equal(20, vector.Length);
            Assert.Equal(4, vector.Nnz);
        }

        [Fact]
        public void Get_ShouldSucced()
        {
            var elements = new[] { (0, 1.0), (5, 2.0), (3, 9.0), (19, 20) };
            SparseVectorD vector = new SparseVectorD(elements, 20);
            Assert.Equal(0.0, vector.Get(2));
            Assert.Equal(9.0, vector.Get(3));
        }


        [Fact]
        public void Set_ShouldSucced()
        {
            var elements = new[] { (0, 1.0), (5, 2.0), (3, 9.0), (19, 20) };
            SparseVectorD vector = new SparseVectorD(elements, 20);
            vector.Set(9, 220);

            Assert.Equal(new[] { 0, 3, 5, 9, 19 }, vector.GetIndices().ToArray());
            Assert.Equal(new[] { 1.0, 9.0, 2.0, 220, 20 }, vector.GetValues().ToArray());
            Assert.Equal(20, vector.Length);
            Assert.Equal(5, vector.Nnz);
        }

        [Fact]
        public void Add_ShouldSucceed()
        {
            var A = new VectorXD(new double[] { 1, 2, 3, 4 }).ToSparse();
            var B = new VectorXD(new double[] { 1, 2, 3, 4 }).ToSparse();
            var addVector = A.Add(B);
            Assert.Equal(new VectorXD(new double[] { 2, 4, 6, 8 }), addVector.ToDense());
        }

        [Fact]
        public void Minus_ShouldSucceed()
        {
            var A = new VectorXD(new double[] { 1, 2, 3, 4 }).ToSparse();
            var B = new VectorXD(new double[] { 1, 2, 3, 4 }).ToSparse();
            var addVector = A.Minus(B);
            Assert.Equal(new VectorXD(new double[] { 0, 0, 0, 0 }), addVector.ToDense());
        }

        [Fact]
        public void Scale_ShouldSucceed()
        {
            var A = new VectorXD(Enumerable.Range(1, 4).Select(n => (double)n).ToArray()).ToSparse();
            var scaledVector = A.Scale(2.0);
            Assert.Equal(new VectorXD(new double[] { 2, 4, 6, 8 }), scaledVector.ToDense());
        }

        [Fact]
        public void ScaleInplace_ShouldSucceed()
        {
            var A = new VectorXD(Enumerable.Range(1, 4).Select(n => (double)n).ToArray()).ToSparse();
            A.ScaleInplace(2.0);
            Assert.Equal(new VectorXD(new double[] { 2, 4, 6, 8 }), A.ToDense());
        }
    }
}
