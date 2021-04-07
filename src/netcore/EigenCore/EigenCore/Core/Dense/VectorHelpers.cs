namespace EigenCore.Core.Dense
{
    public static class VectorHelpers
    {
        private const double DoubleTolerance = 10e-12;

        internal static bool ArraysEqual(double[] array1, double[] array2)
        {
            if (array1.Length == array2.Length)
            {
                for (int i = 0; i < array1.Length; i++)
                {
                    if (array1[i] - array2[i] > DoubleTolerance)
                    {
                        return false;
                    }
                }

                return true;
            }

            return false;
        }
    }
}
