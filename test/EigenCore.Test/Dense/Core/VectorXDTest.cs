using EigenCore.Core.Dense;
using System.Linq;
using Xunit;

namespace EigenCore.Test.Dense.Core
{
    public class VectorXDTest
    {
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
