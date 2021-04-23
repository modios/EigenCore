using System.Runtime.InteropServices;
using System.Security;

namespace EigenCore.Eigen
{
	public static unsafe class ThunkSparseEigen
	{
		internal const string NativeThunkEigenPath = "eigen_core";

		[DllImport(NativeThunkEigenPath), SuppressUnmanagedCodeSecurity]
		public static extern bool ssolve_conjugateGradient_(
		int row,
		int col,
		int nnz,
		[In] int* outerIndex,
		[In] int* innerIndex,
		[In] double* values,
		[In] double* inrhs,
		[In] int size,
		[Out] double* vout);
	}
}