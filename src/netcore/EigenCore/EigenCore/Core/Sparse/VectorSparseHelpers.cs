using EigenCore.Core.Dense;
using System.Linq;

namespace EigenCore.Core.Sparse
{
    public static class VectorSparseHelpers
    {
        public static VectorXD ToDense(this SparseVectorD sparseVectorD)
        {
            return new VectorXD(Enumerable.Range(0, sparseVectorD.Length).Select(i => sparseVectorD.Get(i)).ToArray());
        }

        public static SparseVectorD ToSparse(this VectorXD vector)
        {
            var elements = Enumerable.Range(0, vector.Length)
                .Zip(vector.GetValues().ToArray(), (index, value) => (index, value)).ToArray();
            return new SparseVectorD(elements, vector.Length);
        }

        public static (int[], double[]) SortByIndices(this (int, double)[] inputs)
        {
            var inputList = inputs.ToList();
            inputList.Sort((a, b) => a.Item1.CompareTo(b.Item1));
            return (inputList.Select(input => input.Item1).ToArray(), inputList.Select(input => input.Item2).ToArray());
        }
    }
}
