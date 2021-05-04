using System;

namespace EigenCore.Core.Shared
{
    public static class ArrayHelpers
    {
        private const double DoubleTolerance = 10e-12;

        internal static void Populate<T>(this T[] arr, T value)
        {
            for (int i = 0; i < arr.Length; i++)
            {
                arr[i] = value;
            }
        }

        internal static double ArraysDot(double[] array1, double[] array2)
        {
            double sum = 0;
            if (array1.Length == array2.Length)
            {
                for (int i = 0; i < array1.Length; i++)
                {
                    sum +=array1[i] * array2[i];
                }

                return sum;
            }

            return double.NaN;
        }

        internal static void ArraysScaleInplace(double[] array1, double scalar)
        {
            for (int i = 0; i < array1.Length; i++)
            {
                array1[i] = array1[i] * scalar;
            }
        }

        internal static double[] ArraysScale(double[] array1, double scalar)
        {
            double[] scaledArray = new double[array1.Length];
            for (int i = 0; i < array1.Length; i++)
            {
                scaledArray[i] = array1[i] * scalar;
            }

            return scaledArray;
        }

        internal static double[] ArraysAdd(double[] array1, double[] array2)
        {
            double[] addArray = new double[array1.Length];
            if (array1.Length == array2.Length)
            {
                for (int i = 0; i < array1.Length; i++)
                {
                    addArray[i] = array1[i] + array2[i];
                }

                return addArray;
            }

            return default;
        }

        internal static double[] ArraysMinus(double[] array1, double[] array2)
        {
            double[] minusArray = new double[array1.Length];
            if (array1.Length == array2.Length)
            {
                for (int i = 0; i < array1.Length; i++)
                {
                    minusArray[i] = array1[i] - array2[i];
                }

                return minusArray;
            }

            return default;
        }

        internal static int[] SumArrays(int[] array1, int[] array2)
        {

            if (array1.Length == array2.Length)
            {
                var length = array1.Length;
                var arraySum = new int[length];

                for (int i = 0; i < array1.Length; i++)
                {
                    arraySum[i] = array1[i] + array2[i];
                }

                return arraySum;
            }

            return default;
        }

        internal static bool ArraysEqual(int[] array1, int[] array2)
        {
            if (array1.Length == array2.Length)
            {
                for (int i = 0; i < array1.Length; i++)
                {
                    if (Math.Abs(array1[i] - array2[i]) > 0)
                    {
                        return false;
                    }
                }

                return true;
            }

            return false;
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
