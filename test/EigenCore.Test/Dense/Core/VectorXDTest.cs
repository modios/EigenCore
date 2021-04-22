using EigenCore.Core.Dense;
using System.Linq;
using Xunit;

namespace EigenCore.Test.Dense.Core
{
    public class VectorXDTest
    {
        [Fact]
        public void ArrayConstructor_ShouldSucced()
        {
            VectorXD v = new VectorXD(new double[] { 1, 2, 5, 6 });
            Assert.Equal(new double[] { 1, 2, 5, 6 }, v.GetValues().ToArray());
            Assert.Equal("VectorXD, 4:\n\n1 2 5 6", v.ToString());
        }

        [Fact]
        public void Zeros_ShouldSucceed()
        {
            VectorXD v = VectorXD.Zeros(4);
            Assert.True(v.Max() <= 0.0);
            Assert.True(v.Min() >= 0.0);
            Assert.Equal(4, v.Length);
            Assert.Equal(new VectorXD(new[] { 0.0, 0.0, 0.0, 0.0 }), v);
        }

        [Fact]
        public void Ones_ShouldSucceed()
        {
            VectorXD v = VectorXD.Ones(4);
            Assert.True(v.Max() <= 1.0);
            Assert.True(v.Min() >= 1.0);
            Assert.Equal(4, v.Length);
            Assert.Equal(new VectorXD(new[] { 1.0, 1.0, 1.0, 1.0 }), v);
        }

        [Fact]
        public void Identity_ShouldSucceed()
        {
            VectorXD v = VectorXD.Identity(3, 1);
            Assert.True(v.Max() <= 1.0);
            Assert.True(v.Min() >= 0.0);
            Assert.Equal(3, v.Length);
            Assert.Equal(1, v.Get(1));
            Assert.Equal(0, v.Get(0));
            Assert.Equal(0, v.Get(2));
        }

        [Fact]
        public void Random_ShouldSucceed()
        {
            VectorXD A = VectorXD.Random(10);
            Assert.True(A.Max() <= 1.0);
            Assert.True(A.Min() >= 0.0);
            Assert.Equal(10, A.Length);
        }

        [Fact]
        public void Linspace_ShouldSucceed()
        {
            VectorXD v = VectorXD.Linespace(1, 10, 10);
            Assert.Equal(10.0, v.Max());
            Assert.Equal(1.0, v.Min());
            Assert.Equal(10, v.Length);
            Assert.Equal(new VectorXD(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }), v);
        }

        [Fact]
        public void Dot_ShouldSucceed()
        {
            VectorXD A = new VectorXD(new double[] { 1, 2, 3, 4 });
            VectorXD B = new VectorXD(new double[] { 1, 2, 3, 4 });
            Assert.Equal(30.0, A.Dot(B));
        }

        [InlineData("2 2 1", 3)]
        [InlineData("1 0 0 0", 1)]
        [Theory(Skip = "need to update .so")]
        public void Norm_ShouldSucceed(string valuesString, double expected)
        {
            VectorXD v = new VectorXD(valuesString);
            Assert.Equal(expected, v.Norm());
        }

        [InlineData("2 2 1", 9)]
        [InlineData("1 0 0 0", 1)]
        [Theory(Skip = "need to update .so")]
        public void SquaredNorm_ShouldSucceed(string valuesString, double expected)
        {
            VectorXD v = new VectorXD(valuesString);
            Assert.Equal(expected, v.SquaredNorm());
        }

        [InlineData("2 2 1", 5)]
        [InlineData("2 2 -1", 5)]
        [InlineData("1 0 0 0", 1)]
        [Theory(Skip = "need to update .so")]
        public void Lp1Norm_ShouldSucceed(string valuesString, double expected)
        {
            VectorXD v = new VectorXD(valuesString);
            Assert.Equal(expected, v.Lp1Norm());
        }

        [InlineData("2 2 1", 2)]
        [InlineData("2 2 -3", 3)]
        [InlineData("1 0 0 0", 1)]
        [Theory(Skip = "need to update .so")]
        public void LpInfoNorm_ShouldSucceed(string valuesString, double expected)
        {
            VectorXD v = new VectorXD(valuesString);
            Assert.Equal(expected, v.LpInfoNorm());
        }

        [Fact]
        public void Add_ShouldSucceed()
        {
            VectorXD A = new VectorXD(new double[] { 1, 2, 3, 4 });
            VectorXD B = new VectorXD(new double[] { 1, 2, 3, 4 });
            var addVector = A.Add(B);
            Assert.Equal(new VectorXD(new double[] { 2, 4, 6, 8 }), addVector);
        }

        [Fact]
        public void Scale_ShouldSucceed()
        {
            VectorXD A = new VectorXD(Enumerable.Range(1, 4).Select(n => (double)n).ToArray());
            var scaledVector = A.Scale(2.0);
            Assert.Equal(new VectorXD(new double[] { 2, 4, 6, 8 }), scaledVector);
        }
    }
}
