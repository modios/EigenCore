﻿using System.Runtime.InteropServices;
using System.Security;

namespace EigenCore.Eigen
{
    internal static unsafe class ThunkDenseEigen
    {
        internal const string NativeThunkEigenPath = "eigen_core";
        
        #region Vectors
        
        [DllImport(NativeThunkEigenPath), SuppressUnmanagedCodeSecurity]
        public static extern double ddot_([In] double* firstVector, [In] double* secondVector, int length);

        [DllImport(NativeThunkEigenPath), SuppressUnmanagedCodeSecurity]
        public static extern void dadd_([In] double* firstVector, [In] double* secondVector, int length, [Out] double* vout);

        [DllImport(NativeThunkEigenPath), SuppressUnmanagedCodeSecurity]
        public static extern void dscale_([In] double* firstVector, double scale, int length, [Out] double* vout);

        #endregion Vectors

        #region Matrices
        
        [DllImport(NativeThunkEigenPath), SuppressUnmanagedCodeSecurity]
        public static extern void dmult_([In] double* firstMatrix, int row1, int col1, [In] double* secondMatrix, int row2, int col2, [Out] double* vout);


        [DllImport(NativeThunkEigenPath), SuppressUnmanagedCodeSecurity]
        public static extern void dtransp_([In] double* firstMatrix, int row1, int col1, [Out] double* vout);

        [DllImport(NativeThunkEigenPath), SuppressUnmanagedCodeSecurity]
        public static extern void dmultt_([In] double* firstMatrix, int row1, int col1, [In] double* secondMatrix, int row2, int col2, [Out] double* vout);

        #endregion Matrices
    }
}