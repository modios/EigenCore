using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace EigenCore.Eigen
{
    internal static class EigenDenseUtilities
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
        public static double Norm(ReadOnlySpan<double> firstVector, int length)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstVector))
                {
                    return ThunkDenseEigen.dvnorm_(pfirst, length);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double SquaredNorm(ReadOnlySpan<double> firstVector, int length)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstVector))
                {
                    return ThunkDenseEigen.dvsquared_norm_(pfirst, length);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Lp1Norm(ReadOnlySpan<double> firstVector, int length)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstVector))
                {
                    return ThunkDenseEigen.dvlp1_norm_(pfirst, length);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double LpInfNorm(ReadOnlySpan<double> firstVector, int length)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstVector))
                {
                    return ThunkDenseEigen.dvlpinf_norm_(pfirst, length);
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
        public static void Minus(
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
                            ThunkDenseEigen.dminus_(pfirst, rows1, cols1, pSecond, rows2, cols2, pOut);
                        }
                    }
                }
            }
        }

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
        public static double Norm(
        ReadOnlySpan<double> firstMatrix,
        int rows1,
        int cols1)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    return ThunkDenseEigen.dnorm_(pfirst, rows1, cols1);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double SquaredNorm(
            ReadOnlySpan<double> firstMatrix,
            int rows1,
            int cols1)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    return ThunkDenseEigen.dsquared_norm_(pfirst, rows1, cols1);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Lp1Norm(
            ReadOnlySpan<double> firstMatrix,
            int rows1,
            int cols1)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    return ThunkDenseEigen.dlp1_norm_(pfirst, rows1, cols1);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double LpInfNorm(
            ReadOnlySpan<double> firstMatrix,
            int rows1,
            int cols1)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    return ThunkDenseEigen.dlpinf_norm_(pfirst, rows1, cols1);
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
        public static void Plus(
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
                            ThunkDenseEigen.dxplusa_(pfirst, rows1, cols1, pSecond, rows2, cols2, pOut);
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SelfAdjointEigenSolver(ReadOnlySpan<double> firstMatrix, int size, Span<double> outRealEigenvalues, Span<double> outRealEigenVectors)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    fixed (double* pRealOut = &MemoryMarshal.GetReference(outRealEigenvalues))
                    {
                        fixed (double* pRealVectorOut = &MemoryMarshal.GetReference(outRealEigenVectors))
                        {
                            ThunkDenseEigen.dselfadjoint_eigenvalues_(pfirst, size, pRealOut, pRealVectorOut);

                        }
                    }
                }
            }
        }

        /// <summary>
        /// X = A + A^T
        /// </summary>
        /// <param name="firstMatrix"></param>
        /// <param name="size"></param>
        /// <param name="outMatrix"></param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void PlusT(ReadOnlySpan<double> firstMatrix, int size, Span<double> outMatrix)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    fixed (double* pOut = &MemoryMarshal.GetReference(outMatrix))
                    {
                        ThunkDenseEigen.dxplusxt_(pfirst, size, pOut);
                    }
                }
            }
        }

        /// <summary>
        /// X = A * A^T
        /// </summary>
        /// <param name="firstMatrix"></param>
        /// <param name="size"></param>
        /// <param name="outMatrix"></param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void MultT(ReadOnlySpan<double> firstMatrix, int rows, int cols, Span<double> outMatrix)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    fixed (double* pOut = &MemoryMarshal.GetReference(outMatrix))
                    {
                        ThunkDenseEigen.da_multt_(pfirst, rows, cols, pOut);
                    }
                }
            }
        }

        /// <summary>
        /// X = A^T * A
        /// </summary>
        /// <param name="firstMatrix"></param>
        /// <param name="size"></param>
        /// <param name="outMatrix"></param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void TMult(ReadOnlySpan<double> firstMatrix, int rows, int cols, Span<double> outMatrix)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    fixed (double* pOut = &MemoryMarshal.GetReference(outMatrix))
                    {
                        ThunkDenseEigen.da_tmult_(pfirst, rows, cols, pOut);
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SVD(ReadOnlySpan<double> firstMatrix, int rows1, int cols1,
            Span<double> uout,
            Span<double> sout,
            Span<double> vout)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    fixed (double* pUOut = &MemoryMarshal.GetReference(uout))
                    {
                        fixed (double* pSOut = &MemoryMarshal.GetReference(sout))
                        {
                            fixed (double* pVOut = &MemoryMarshal.GetReference(vout))
                            {
                                ThunkDenseEigen.dsvd_(pfirst, rows1, cols1, pUOut, pSOut, pVOut);
                            }

                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SVDLeastSquares(ReadOnlySpan<double> firstMatrix, int rows1, int cols1,
         ReadOnlySpan<double> rhs,
         Span<double> vout)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    fixed (double* prhs = &MemoryMarshal.GetReference(rhs))
                    {
                        fixed (double* pVOut = &MemoryMarshal.GetReference(vout))
                        {
                            ThunkDenseEigen.dsvd_leastsquares_(pfirst, rows1, cols1, prhs, pVOut);
                        }

                    }
                }
            }
        }



        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SVDBdcSvd(ReadOnlySpan<double> firstMatrix, int rows1, int cols1,
        Span<double> uout,
        Span<double> sout,
        Span<double> vout)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    fixed (double* pUOut = &MemoryMarshal.GetReference(uout))
                    {
                        fixed (double* pSOut = &MemoryMarshal.GetReference(sout))
                        {
                            fixed (double* pVOut = &MemoryMarshal.GetReference(vout))
                            {
                                ThunkDenseEigen.dsvd_bdcSvd_(pfirst, rows1, cols1, pUOut, pSOut, pVOut);
                            }

                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SVDLeastSquaresBdcSvd(ReadOnlySpan<double> firstMatrix, int rows1, int cols1,
         ReadOnlySpan<double> rhs,
         Span<double> vout)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    fixed (double* prhs = &MemoryMarshal.GetReference(rhs))
                    {
                        fixed (double* pVOut = &MemoryMarshal.GetReference(vout))
                        {
                            ThunkDenseEigen.dsvd_bdcSvd__leastsquares_(pfirst, rows1, cols1, prhs, pVOut);
                        }

                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void NormalEquationsLeastSquares(ReadOnlySpan<double> firstMatrix, int rows1, int cols1,
            ReadOnlySpan<double> rhs,
            Span<double> vout)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    fixed (double* prhs = &MemoryMarshal.GetReference(rhs))
                    {
                        fixed (double* pVOut = &MemoryMarshal.GetReference(vout))
                        {
                            ThunkDenseEigen.dnormal_equations__leastsquares_(pfirst, rows1, cols1, prhs, pVOut);
                        }

                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SolveColPivHouseholderQr(ReadOnlySpan<double> firstMatrix,
            int rows1,
            int cols1,
            ReadOnlySpan<double> rhs,
            Span<double> vout)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    fixed (double* prhs = &MemoryMarshal.GetReference(rhs))
                    {
                        fixed (double* pVOut = &MemoryMarshal.GetReference(vout))
                        {
                            ThunkDenseEigen.dsolve_colPivHouseholderQr_(pfirst, rows1, cols1, prhs, pVOut);
                        }

                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SolveFullPivLU(ReadOnlySpan<double> firstMatrix,
        int rows1,
        int cols1,
        ReadOnlySpan<double> rhs,
        Span<double> vout)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    fixed (double* prhs = &MemoryMarshal.GetReference(rhs))
                    {
                        fixed (double* pVOut = &MemoryMarshal.GetReference(vout))
                        {
                            ThunkDenseEigen.dsolve_fullPivLu_(pfirst, rows1, cols1, prhs, pVOut);
                        }

                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SolvePartialPivLU(ReadOnlySpan<double> firstMatrix,
            int rows1,
            int cols1,
            ReadOnlySpan<double> rhs,
            Span<double> vout)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    fixed (double* prhs = &MemoryMarshal.GetReference(rhs))
                    {
                        fixed (double* pVOut = &MemoryMarshal.GetReference(vout))
                        {
                            ThunkDenseEigen.dsolve_partialPivLU_(pfirst, rows1, cols1, prhs, pVOut);
                        }

                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SolveLLT(ReadOnlySpan<double> firstMatrix,
                int rows1,
                int cols1,
                ReadOnlySpan<double> rhs,
                Span<double> vout)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    fixed (double* prhs = &MemoryMarshal.GetReference(rhs))
                    {
                        fixed (double* pVOut = &MemoryMarshal.GetReference(vout))
                        {
                            ThunkDenseEigen.dsolve_llt_(pfirst, rows1, cols1, prhs, pVOut);
                        }

                    }
                }
            }
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SolveLDLT(ReadOnlySpan<double> firstMatrix,
        int rows1,
        int cols1,
        ReadOnlySpan<double> rhs,
        Span<double> vout)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    fixed (double* prhs = &MemoryMarshal.GetReference(rhs))
                    {
                        fixed (double* pVOut = &MemoryMarshal.GetReference(vout))
                        {
                            ThunkDenseEigen.dsolve_ldlt_(pfirst, rows1, cols1, prhs, pVOut);
                        }

                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Determinant(ReadOnlySpan<double> firstMatrix, int rows1, int cols1)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    return ThunkDenseEigen.ddeterminant_(pfirst, rows1, cols1);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Determinant(ReadOnlySpan<double> firstMatrix, int rows1, int cols1, Span<double> mout)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    fixed (double* pmout = &MemoryMarshal.GetReference(mout))
                    {
                        ThunkDenseEigen.dinverse_(pfirst, rows1, cols1, pmout);
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double AbsoluteError(ReadOnlySpan<double> firstMatrix, int rows1, int cols1, ReadOnlySpan<double> rhs, ReadOnlySpan<double> x)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    fixed (double* pX = &MemoryMarshal.GetReference(x))
                    {
                        fixed (double* pRhs = &MemoryMarshal.GetReference(rhs))
                        {
                            return ThunkDenseEigen.dabsolute_error_(pfirst, rows1, cols1, pRhs, pX);
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double RelativeError(ReadOnlySpan<double> firstMatrix, int rows1, int cols1, ReadOnlySpan<double> rhs, ReadOnlySpan<double> x)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    fixed (double* pX = &MemoryMarshal.GetReference(x))
                    {
                        fixed (double* pRhs = &MemoryMarshal.GetReference(rhs))
                        {
                            return ThunkDenseEigen.drelative_error_(pfirst, rows1, cols1, pRhs, pX);
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void HouseholderQR(ReadOnlySpan<double> firstMatrix,
        int rows1,
        int cols1,
        Span<double> q,
        Span<double> r)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    fixed (double* pQ = &MemoryMarshal.GetReference(q))
                    {
                        fixed (double* pR = &MemoryMarshal.GetReference(r))
                        {
                            ThunkDenseEigen.dhouseholderQR_(pfirst, rows1, cols1, pQ, pR);
                        }

                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ColPivHouseholderQR(ReadOnlySpan<double> firstMatrix,
                int rows1,
                int cols1,
                Span<double> q,
                Span<double> r,
                Span<double> p)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    fixed (double* pQ = &MemoryMarshal.GetReference(q))
                    {
                        fixed (double* pR = &MemoryMarshal.GetReference(r))
                        {
                            fixed (double* pP = &MemoryMarshal.GetReference(p))
                            {
                                ThunkDenseEigen.dcolPivHouseholderQR_(pfirst, rows1, cols1, pQ, pR, pP);
                            }
                        }

                    }
                }
            }
        }

        public static void FullPivLU(ReadOnlySpan<double> firstMatrix,
               int rows1,
               int cols1,
               Span<double> l,
               Span<double> u,
               Span<double> p,
               Span<double> q)
        {
            unsafe
            {
                fixed (double* pfirst = &MemoryMarshal.GetReference(firstMatrix))
                {
                    fixed (double* pL = &MemoryMarshal.GetReference(l))
                    {
                        fixed (double* pU = &MemoryMarshal.GetReference(u))
                        {
                            fixed (double* pP = &MemoryMarshal.GetReference(p))
                            {
                                fixed (double* pQ = &MemoryMarshal.GetReference(q))
                                {
                                    ThunkDenseEigen.dfullPivLU_(pfirst, rows1, cols1, pL, pU, pP, pQ);
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
