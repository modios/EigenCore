using EigenCore.Core.Dense;
using Xunit;

namespace EigenCore.Test.Dense.Core
{
    public class VectorDenseBaseTest
    {
        [Fact]
        public void ToString_ShouldSucceed()
        {
            var A = new VectorXD(new double[] { 1, 3, 1 });
            Assert.Equal("VectorXD, 3:\n\n1 3 1", A.ToString());

            var B = new VectorXD(new double[] { 1.3423432, 3.234324, 3243241 });
            Assert.Equal("VectorXD, 3:\n\n1.34 3.23 3.24E+06", B.ToString());

            var C = VectorXD.Linespace(0, 99, 100);
            Assert.Equal("VectorXD, 100:\n\n0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ...", C.ToString());
        }
    }
}
