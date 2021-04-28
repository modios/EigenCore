using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace EigenCore.Eigen
{
    internal class EigenSparseUtilities
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool SolveConjugateGradient(
        int rows,
        int cols,
        int nnz,
        int maxIterations,
        double tolerance,
        ReadOnlySpan<int> outerIndex,
        ReadOnlySpan<int> innerIndex,
        ReadOnlySpan<double> values,
        ReadOnlySpan<double> rhs,
        int size,
        Span<double> vout,
        out int iterations,
        out double error)
        {
            unsafe
            {
                int iterationsOut;
                double errorOut;
                fixed (int* pOuterIndex = &MemoryMarshal.GetReference(outerIndex))
                {
                    fixed (int* pInnerIndex = &MemoryMarshal.GetReference(innerIndex))
                    {
                        fixed (double* pValues = &MemoryMarshal.GetReference(values))
                        {
                            fixed (double* pRhs = &MemoryMarshal.GetReference(rhs))
                            {
                                fixed (double* pVOut = &MemoryMarshal.GetReference(vout))
                                {
                                    bool result = ThunkSparseEigen.ssolve_conjugateGradient_(rows, cols, nnz, maxIterations, tolerance, pOuterIndex, pInnerIndex, pValues, pRhs, size, pVOut, &iterationsOut, &errorOut);
                                    iterations = iterationsOut;
                                    error = errorOut;
                                    return result;
                                }
                            }
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool SolveBiCGSTAB(
            int rows,
            int cols,
            int nnz,
            int maxIterations,
            double tolerance,
            ReadOnlySpan<int> outerIndex,
            ReadOnlySpan<int> innerIndex,
            ReadOnlySpan<double> values,
            ReadOnlySpan<double> rhs,
            int size,
            Span<double> vout,
            out int iterations,
            out double error)
        {
            unsafe
            {
                int iterationsOut;
                double errorOut;
                fixed (int* pOuterIndex = &MemoryMarshal.GetReference(outerIndex))
                {
                    fixed (int* pInnerIndex = &MemoryMarshal.GetReference(innerIndex))
                    {
                        fixed (double* pValues = &MemoryMarshal.GetReference(values))
                        {
                            fixed (double* pRhs = &MemoryMarshal.GetReference(rhs))
                            {
                                fixed (double* pVOut = &MemoryMarshal.GetReference(vout))
                                {
                                    bool result = ThunkSparseEigen.ssolve_biCGSTAB_(rows, cols, nnz, maxIterations, tolerance, pOuterIndex, pInnerIndex, pValues, pRhs, size, pVOut, &iterationsOut, &errorOut);
                                    iterations = iterationsOut;
                                    error = errorOut;
                                    return result;
                                }
                            }
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool SolveLeastSquaresConjugateGradient(
            int rows,
            int cols,
            int nnz,
            int maxIterations,
            double tolerance,
            ReadOnlySpan<int> outerIndex,
            ReadOnlySpan<int> innerIndex,
            ReadOnlySpan<double> values,
            ReadOnlySpan<double> rhs,
            int size,
            Span<double> vout,
            out int iterations,
            out double error)
        {
            unsafe
            {
                int iterationsOut;
                double errorOut;
                fixed (int* pOuterIndex = &MemoryMarshal.GetReference(outerIndex))
                {
                    fixed (int* pInnerIndex = &MemoryMarshal.GetReference(innerIndex))
                    {
                        fixed (double* pValues = &MemoryMarshal.GetReference(values))
                        {
                            fixed (double* pRhs = &MemoryMarshal.GetReference(rhs))
                            {
                                fixed (double* pVOut = &MemoryMarshal.GetReference(vout))
                                {
                                    bool result = ThunkSparseEigen.ssolve_LeastSquaresConjugateGradient_(rows, cols, nnz, maxIterations, tolerance, pOuterIndex, pInnerIndex, pValues, pRhs, size, pVOut, &iterationsOut, &errorOut);
                                    iterations = iterationsOut;
                                    error = errorOut;
                                    return result;
                                }
                            }
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool SolveGMRES(
            int rows,
            int cols,
            int nnz,
            int maxIterations,
            double tolerance,
            ReadOnlySpan<int> outerIndex,
            ReadOnlySpan<int> innerIndex,
            ReadOnlySpan<double> values,
            ReadOnlySpan<double> rhs,
            int size,
            Span<double> vout,
            out int iterations,
            out double error)
        {
            unsafe
            {
                int iterationsOut;
                double errorOut;
                fixed (int* pOuterIndex = &MemoryMarshal.GetReference(outerIndex))
                {
                    fixed (int* pInnerIndex = &MemoryMarshal.GetReference(innerIndex))
                    {
                        fixed (double* pValues = &MemoryMarshal.GetReference(values))
                        {
                            fixed (double* pRhs = &MemoryMarshal.GetReference(rhs))
                            {
                                fixed (double* pVOut = &MemoryMarshal.GetReference(vout))
                                {
                                    bool result = ThunkSparseEigen.ssolve_GMRES_(rows, cols, nnz, maxIterations, tolerance, pOuterIndex, pInnerIndex, pValues, pRhs, size, pVOut, &iterationsOut, &errorOut);
                                    iterations = iterationsOut;
                                    error = errorOut;
                                    return result;
                                }
                            }
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool SolveMINRES(
            int rows,
            int cols,
            int nnz,
            int maxIterations,
            double tolerance,
            ReadOnlySpan<int> outerIndex,
            ReadOnlySpan<int> innerIndex,
            ReadOnlySpan<double> values,
            ReadOnlySpan<double> rhs,
            int size,
            Span<double> vout,
            out int iterations,
            out double error)
        {
            unsafe
            {
                int iterationsOut;
                double errorOut;
                fixed (int* pOuterIndex = &MemoryMarshal.GetReference(outerIndex))
                {
                    fixed (int* pInnerIndex = &MemoryMarshal.GetReference(innerIndex))
                    {
                        fixed (double* pValues = &MemoryMarshal.GetReference(values))
                        {
                            fixed (double* pRhs = &MemoryMarshal.GetReference(rhs))
                            {
                                fixed (double* pVOut = &MemoryMarshal.GetReference(vout))
                                {
                                    bool result = ThunkSparseEigen.ssolve_MINRES_(rows, cols, nnz, maxIterations, tolerance, pOuterIndex, pInnerIndex, pValues, pRhs, size, pVOut, &iterationsOut, &errorOut);
                                    iterations = iterationsOut;
                                    error = errorOut;
                                    return result;
                                }
                            }
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool SolveDGMRES(
            int rows,
            int cols,
            int nnz,
            int maxIterations,
            double tolerance,
            ReadOnlySpan<int> outerIndex,
            ReadOnlySpan<int> innerIndex,
            ReadOnlySpan<double> values,
            ReadOnlySpan<double> rhs,
            int size,
            Span<double> vout,
            out int iterations,
            out double error)
        {
            unsafe
            {
                int iterationsOut;
                double errorOut;
                fixed (int* pOuterIndex = &MemoryMarshal.GetReference(outerIndex))
                {
                    fixed (int* pInnerIndex = &MemoryMarshal.GetReference(innerIndex))
                    {
                        fixed (double* pValues = &MemoryMarshal.GetReference(values))
                        {
                            fixed (double* pRhs = &MemoryMarshal.GetReference(rhs))
                            {
                                fixed (double* pVOut = &MemoryMarshal.GetReference(vout))
                                {
                                    bool result = ThunkSparseEigen.ssolve_DGMRES_(rows, cols, nnz, maxIterations, tolerance, pOuterIndex, pInnerIndex, pValues, pRhs, size, pVOut, &iterationsOut, &errorOut);
                                    iterations = iterationsOut;
                                    error = errorOut;
                                    return result;
                                }
                            }
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ADD(
        int rows,
        int cols,
        int nnz1,
        ReadOnlySpan<int> outerIndex1,
        ReadOnlySpan<int> innerIndex1,
        ReadOnlySpan<double> values1,
        int nnz2,
        ReadOnlySpan<int> outerIndex2,
        ReadOnlySpan<int> innerIndex2,
        ReadOnlySpan<double> values2,
        Span<int> outerIndex,
        Span<int> innerIndex,
        Span<double> values,
        out int nnz)
        {
            unsafe
            {

                fixed (int* pOuterIndex = &MemoryMarshal.GetReference(outerIndex))
                {
                    fixed (int* pInnerIndex = &MemoryMarshal.GetReference(innerIndex))
                    {
                        fixed (double* pValues = &MemoryMarshal.GetReference(values))
                        {
                            fixed (int* pOuterIndex1 = &MemoryMarshal.GetReference(outerIndex1))
                            {
                                fixed (int* pInnerIndex1 = &MemoryMarshal.GetReference(innerIndex1))
                                {
                                    fixed (double* pValues1 = &MemoryMarshal.GetReference(values1))
                                    {
                                        fixed (int* pOuterIndex2 = &MemoryMarshal.GetReference(outerIndex2))
                                        {
                                            fixed (int* pInnerIndex2 = &MemoryMarshal.GetReference(innerIndex2))
                                            {
                                                fixed (double* pValues2 = &MemoryMarshal.GetReference(values2))
                                                {
                                                    int outNnz;
                                                    ThunkSparseEigen.sadd_(rows, cols,
                                                    nnz1, pOuterIndex1, pInnerIndex1, pValues1,
                                                    nnz2, pOuterIndex2, pInnerIndex2, pValues2,
                                                    &outNnz, pOuterIndex, pInnerIndex, pValues);

                                                    nnz = outNnz;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Minus(
               int rows,
               int cols,
               int nnz1,
               ReadOnlySpan<int> outerIndex1,
               ReadOnlySpan<int> innerIndex1,
               ReadOnlySpan<double> values1,
               int nnz2,
               ReadOnlySpan<int> outerIndex2,
               ReadOnlySpan<int> innerIndex2,
               ReadOnlySpan<double> values2,
               Span<int> outerIndex,
               Span<int> innerIndex,
               Span<double> values,
               out int nnz)
        {
            unsafe
            {

                fixed (int* pOuterIndex = &MemoryMarshal.GetReference(outerIndex))
                {
                    fixed (int* pInnerIndex = &MemoryMarshal.GetReference(innerIndex))
                    {
                        fixed (double* pValues = &MemoryMarshal.GetReference(values))
                        {
                            fixed (int* pOuterIndex1 = &MemoryMarshal.GetReference(outerIndex1))
                            {
                                fixed (int* pInnerIndex1 = &MemoryMarshal.GetReference(innerIndex1))
                                {
                                    fixed (double* pValues1 = &MemoryMarshal.GetReference(values1))
                                    {
                                        fixed (int* pOuterIndex2 = &MemoryMarshal.GetReference(outerIndex2))
                                        {
                                            fixed (int* pInnerIndex2 = &MemoryMarshal.GetReference(innerIndex2))
                                            {
                                                fixed (double* pValues2 = &MemoryMarshal.GetReference(values2))
                                                {
                                                    int outNnz;
                                                    ThunkSparseEigen.sminus_(rows, cols,
                                                    nnz1, pOuterIndex1, pInnerIndex1, pValues1,
                                                    nnz2, pOuterIndex2, pInnerIndex2, pValues2,
                                                    &outNnz, pOuterIndex, pInnerIndex, pValues);

                                                    nnz = outNnz;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Mult(
              int rows,
              int cols,
              int nnz1,
              ReadOnlySpan<int> outerIndex1,
              ReadOnlySpan<int> innerIndex1,
              ReadOnlySpan<double> values1,
              int nnz2,
              ReadOnlySpan<int> outerIndex2,
              ReadOnlySpan<int> innerIndex2,
              ReadOnlySpan<double> values2,
              Span<int> outerIndex,
              Span<int> innerIndex,
              Span<double> values,
              out int nnz)
        {
            unsafe
            {

                fixed (int* pOuterIndex = &MemoryMarshal.GetReference(outerIndex))
                {
                    fixed (int* pInnerIndex = &MemoryMarshal.GetReference(innerIndex))
                    {
                        fixed (double* pValues = &MemoryMarshal.GetReference(values))
                        {
                            fixed (int* pOuterIndex1 = &MemoryMarshal.GetReference(outerIndex1))
                            {
                                fixed (int* pInnerIndex1 = &MemoryMarshal.GetReference(innerIndex1))
                                {
                                    fixed (double* pValues1 = &MemoryMarshal.GetReference(values1))
                                    {
                                        fixed (int* pOuterIndex2 = &MemoryMarshal.GetReference(outerIndex2))
                                        {
                                            fixed (int* pInnerIndex2 = &MemoryMarshal.GetReference(innerIndex2))
                                            {
                                                fixed (double* pValues2 = &MemoryMarshal.GetReference(values2))
                                                {
                                                    int outNnz;
                                                    ThunkSparseEigen.smult_(rows, cols,
                                                    nnz1, pOuterIndex1, pInnerIndex1, pValues1,
                                                    nnz2, pOuterIndex2, pInnerIndex2, pValues2,
                                                    &outNnz, pOuterIndex, pInnerIndex, pValues);

                                                    nnz = outNnz;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Mult(
              int row,
              int col,
              int nnz,
              ReadOnlySpan<int> outerIndex,
              ReadOnlySpan<int> innerIndex,
              ReadOnlySpan<double> values,
              ReadOnlySpan<double> vector,
              int length,
              Span<double> outMatrix
            )
        {
            unsafe
            {

                fixed (int* pOuterIndex = &MemoryMarshal.GetReference(outerIndex))
                {
                    fixed (int* pInnerIndex = &MemoryMarshal.GetReference(innerIndex))
                    {
                        fixed (double* pValues = &MemoryMarshal.GetReference(values))
                        {
                            fixed (double* pSecond = &MemoryMarshal.GetReference(vector))
                            {
                                fixed (double* pOut = &MemoryMarshal.GetReference(outMatrix))
                                {
                                    ThunkSparseEigen.smultv_(row, col, nnz, pOuterIndex, pInnerIndex, pValues, pSecond, length, pOut);
                                }
                            }

                        }
                    }
                }
            }
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Transpose(
            int rows,
            int cols,
            int nnz,
            ReadOnlySpan<int> outerIndex1,
            ReadOnlySpan<int> innerIndex1,
            ReadOnlySpan<double> values1,
            Span<int> outerIndex,
            Span<int> innerIndex,
            Span<double> values)
        {
            unsafe
            {

                fixed (int* pOuterIndex = &MemoryMarshal.GetReference(outerIndex))
                {
                    fixed (int* pInnerIndex = &MemoryMarshal.GetReference(innerIndex))
                    {
                        fixed (double* pValues = &MemoryMarshal.GetReference(values))
                        {
                            fixed (int* pOuterIndex1 = &MemoryMarshal.GetReference(outerIndex1))
                            {
                                fixed (int* pInnerIndex1 = &MemoryMarshal.GetReference(innerIndex1))
                                {
                                    fixed (double* pValues1 = &MemoryMarshal.GetReference(values1))
                                    {
                                        ThunkSparseEigen.stranspose_(rows, cols,
                                        nnz, pOuterIndex1, pInnerIndex1, pValues1,
                                        pOuterIndex, pInnerIndex, pValues);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SolveSimplicialLDLT(
            int rows,
            int cols,
            int nnz,
            ReadOnlySpan<int> outerIndex,
            ReadOnlySpan<int> innerIndex,
            ReadOnlySpan<double> values,
            ReadOnlySpan<double> rhs,
            int size,
            Span<double> vout)
        {
            unsafe
            {
                fixed (int* pOuterIndex = &MemoryMarshal.GetReference(outerIndex))
                {
                    fixed (int* pInnerIndex = &MemoryMarshal.GetReference(innerIndex))
                    {
                        fixed (double* pValues = &MemoryMarshal.GetReference(values))
                        {
                            fixed (double* pRhs = &MemoryMarshal.GetReference(rhs))
                            {
                                fixed (double* pVOut = &MemoryMarshal.GetReference(vout))
                                {
                                    ThunkSparseEigen.ssolve_simplicialLDLT_(rows, cols, nnz, pOuterIndex, pInnerIndex, pValues, pRhs, size, pVOut);
                                }
                            }
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SolveSparseLU(
            int rows,
            int cols,
            int nnz,
            ReadOnlySpan<int> outerIndex,
            ReadOnlySpan<int> innerIndex,
            ReadOnlySpan<double> values,
            ReadOnlySpan<double> rhs,
            int size,
            Span<double> vout)
        {
            unsafe
            {
                fixed (int* pOuterIndex = &MemoryMarshal.GetReference(outerIndex))
                {
                    fixed (int* pInnerIndex = &MemoryMarshal.GetReference(innerIndex))
                    {
                        fixed (double* pValues = &MemoryMarshal.GetReference(values))
                        {
                            fixed (double* pRhs = &MemoryMarshal.GetReference(rhs))
                            {
                                fixed (double* pVOut = &MemoryMarshal.GetReference(vout))
                                {
                                    ThunkSparseEigen.ssolve_sparseLU_(rows, cols, nnz, pOuterIndex, pInnerIndex, pValues, pRhs, size, pVOut);
                                }
                            }
                        }
                    }
                }
            }
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SolveSparseQR(
            int rows,
            int cols,
            int nnz,
            ReadOnlySpan<int> outerIndex,
            ReadOnlySpan<int> innerIndex,
            ReadOnlySpan<double> values,
            ReadOnlySpan<double> rhs,
            int size,
            Span<double> vout)
        {
            unsafe
            {
                fixed (int* pOuterIndex = &MemoryMarshal.GetReference(outerIndex))
                {
                    fixed (int* pInnerIndex = &MemoryMarshal.GetReference(innerIndex))
                    {
                        fixed (double* pValues = &MemoryMarshal.GetReference(values))
                        {
                            fixed (double* pRhs = &MemoryMarshal.GetReference(rhs))
                            {
                                fixed (double* pVOut = &MemoryMarshal.GetReference(vout))
                                {
                                    ThunkSparseEigen.ssolve_sparseQR_(rows, cols, nnz, pOuterIndex, pInnerIndex, pValues, pRhs, size, pVOut);
                                }
                            }
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SolveSimplicialLLT(
            int rows,
            int cols,
            int nnz,
            ReadOnlySpan<int> outerIndex,
            ReadOnlySpan<int> innerIndex,
            ReadOnlySpan<double> values,
            ReadOnlySpan<double> rhs,
            int size,
            Span<double> vout)
        {
            unsafe
            {
                fixed (int* pOuterIndex = &MemoryMarshal.GetReference(outerIndex))
                {
                    fixed (int* pInnerIndex = &MemoryMarshal.GetReference(innerIndex))
                    {
                        fixed (double* pValues = &MemoryMarshal.GetReference(values))
                        {
                            fixed (double* pRhs = &MemoryMarshal.GetReference(rhs))
                            {
                                fixed (double* pVOut = &MemoryMarshal.GetReference(vout))
                                {
                                    ThunkSparseEigen.ssolve_simplicialLLT_(rows, cols, nnz, pOuterIndex, pInnerIndex, pValues, pRhs, size, pVOut);
                                }
                            }
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void LeastSquaresLU(
            int rows,
            int cols,
            int nnz,
            ReadOnlySpan<int> outerIndex,
            ReadOnlySpan<int> innerIndex,
            ReadOnlySpan<double> values,
            ReadOnlySpan<double> rhs,
            int size,
            Span<double> vout)
        {
            unsafe
            {
                fixed (int* pOuterIndex = &MemoryMarshal.GetReference(outerIndex))
                {
                    fixed (int* pInnerIndex = &MemoryMarshal.GetReference(innerIndex))
                    {
                        fixed (double* pValues = &MemoryMarshal.GetReference(values))
                        {
                            fixed (double* pRhs = &MemoryMarshal.GetReference(rhs))
                            {
                                fixed (double* pVOut = &MemoryMarshal.GetReference(vout))
                                {
                                    ThunkSparseEigen.snormal_equations__leastsquares_sparselu_(rows, cols, nnz, pOuterIndex, pInnerIndex, pValues, pRhs, size, pVOut);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}