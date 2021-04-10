using EigenCore.Core.Dense;
using System.Linq;
using Xunit;

namespace EigenCore.Test.Dense.Core
{
    public class VectorXDTest
    {
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
            VectorXD A = VectorXD.Linespace(1, 10, 10);
            Assert.Equal(10.0, A.Max());
            Assert.Equal(1.0, A.Min());
            Assert.Equal(10, A.Length);
            Assert.Equal(
                new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, 
                A.GetValues().ToArray());
        }

        [Fact]
        public void Dot_ShouldSucceed()
        {
            VectorXD A = new VectorXD(Enumerable.Range(1, 4).Select(n => (double)n).ToArray());
            VectorXD B = new VectorXD(Enumerable.Range(1, 4).Select(n => (double)n).ToArray());
            Assert.Equal(30.0, A.Dot(B));
        }

        [Fact]
        public void Add_ShouldSucceed()
        {
            VectorXD A = new VectorXD(Enumerable.Range(1, 4).Select(n => (double)n).ToArray());
            VectorXD B = new VectorXD(Enumerable.Range(1, 4).Select(n => (double)n).ToArray());
            var addVector = A.Add(B);
            Assert.Equal(new double[] { 2,4,6,8}, addVector.GetValues().ToArray());
        }

        [Fact]
        public void Scale_ShouldSucceed()
        {
            VectorXD A = new VectorXD(Enumerable.Range(1, 4).Select(n => (double)n).ToArray());
            var scaledVector = A.Scale(2.0);
            Assert.Equal(new double[] { 2, 4, 6, 8 }, scaledVector.GetValues().ToArray());
        }
    }
}
