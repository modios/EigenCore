using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace EigenCore.Eigen
{
    internal class EigenSparseUtilities
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool ConjugateGradient(
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
                                   return ThunkSparseEigen.ssolve_conjugateGradient_(rows, cols, nnz, pOuterIndex, pInnerIndex, pValues, pRhs, size, pVOut);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
