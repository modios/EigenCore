using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace EigenCore.Eigen
{
    internal class EigenDenseUtilities
    {
        #region Vectors

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Dot(ReadOnlySpan<double> firstVector, ReadOnlySpan<double> secondVector, int length)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstVector))
                {
                    fixed (double* pSecond = &MemoryMarshal.GetReference(secondVector))
                    {
                        return ThunkDenseEigen.ddot_(pfirst, pSecond, length);
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Add(ReadOnlySpan<double> firstVector, ReadOnlySpan<double> secondVector, int length, Span<double> outVector)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstVector))
                {
                    fixed (double* pSecond = &MemoryMarshal.GetReference(secondVector))
                    {
                        fixed (double* pOut = &MemoryMarshal.GetReference(outVector))
                        {
                            ThunkDenseEigen.dadd_(pfirst, pSecond, length, pOut);
                        }
                    }
                }
            }
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Scale(ReadOnlySpan<double> vector, double scalar, int length, Span<double> outVector)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(vector))
                {
                    fixed (double* pOut = &MemoryMarshal.GetReference(outVector))
                    {
                        ThunkDenseEigen.dscale_(pfirst, scalar, length, pOut);
                    }
                }
            }
        }

        #endregion Vectors

        #region Matrices

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Mult(
                ReadOnlySpan<double> firstMatrix, int rows1, int cols1,
                ReadOnlySpan<double> secondMatrix, int rows2, int cols2,
                Span<double> outMatrix)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    fixed (double* pSecond = &MemoryMarshal.GetReference(secondMatrix))
                    {
                        fixed (double* pOut = &MemoryMarshal.GetReference(outMatrix))
                        {
                            ThunkDenseEigen.dmult_(pfirst, rows1, cols1, pSecond, rows2, cols2, pOut);
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Transpose(
                ReadOnlySpan<double> firstMatrix,
                int rows1,
                int cols1,
                Span<double> outMatrix)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    fixed (double* pOut = &MemoryMarshal.GetReference(outMatrix))
                    {
                        ThunkDenseEigen.dtransp_(pfirst, rows1, cols1, pOut);
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void MultT(
                ReadOnlySpan<double> firstMatrix, int rows1, int cols1,
                ReadOnlySpan<double> secondMatrix, int rows2, int cols2,
                Span<double> outMatrix)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    fixed (double* pSecond = &MemoryMarshal.GetReference(secondMatrix))
                    {
                        fixed (double* pOut = &MemoryMarshal.GetReference(outMatrix))
                        {
                            ThunkDenseEigen.dmultt_(pfirst, rows1, cols1, pSecond, rows2, cols2, pOut);
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Mult(
        ReadOnlySpan<double> firstMatrix, int rows1, int cols1,
        ReadOnlySpan<double> vector, int length,
        Span<double> outMatrix)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    fixed (double* pSecond = &MemoryMarshal.GetReference(vector))
                    {
                        fixed (double* pOut = &MemoryMarshal.GetReference(outMatrix))
                        {
                            ThunkDenseEigen.dmultv_(pfirst, rows1, cols1, pSecond, length, pOut);
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Trace(ReadOnlySpan<double> firstMatrix, int rows1, int cols1)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    return ThunkDenseEigen.dtrace_(pfirst, rows1, cols1);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void EigenSolver(
            ReadOnlySpan<double> firstMatrix, int size,
            Span<double> outRealEigenvalues,
            Span<double> outImagEigenValue,
            Span<double> outRealEigenVectors,
            Span<double> outImagEigenVectors)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    fixed (double* pRealOut = &MemoryMarshal.GetReference(outRealEigenvalues))
                    {
                        fixed (double* pImageOut = &MemoryMarshal.GetReference(outImagEigenValue))
                        {
                            fixed (double* pRealVectorOut = &MemoryMarshal.GetReference(outRealEigenVectors))
                            {
                                fixed (double* pImageVectorOut = &MemoryMarshal.GetReference(outImagEigenVectors))
                                {
                                    ThunkDenseEigen.deigenvalues_(pfirst, size, pRealOut, pImageOut, pRealVectorOut, pImageVectorOut);
                                }
                            }
                        }
                    }
                }
            }
        }

        #endregion Matrices
    }
}





