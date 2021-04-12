using System;

namespace EigenCore.Core.Dense
{
    public static class VectorHelpers
    {
        private const double DoubleTolerance = 10e-12;

        internal static void Populate<T>(this T[] arr, T value)
        {
            for (int i = 0; i < arr.Length; i++)
            {
                arr[i] = value;
            }
        }

        internal static bool ArraysEqual(double[] array1, double[] array2)
        {
            if (array1.Length == array2.Length)
            {
                for (int i = 0; i < array1.Length; i++)
                {
                    if (Math.Abs(array1[i] - array2[i]) > DoubleTolerance)
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
