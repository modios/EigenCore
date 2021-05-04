using System;

namespace EigenCore.Core.Sparse
{
    public abstract class VBufferSparse<T>
    {
        protected readonly T[] _values;

        protected readonly int[] _indices;

        public readonly int Length;

        public int Nnz { get; protected set; }

        public ReadOnlySpan<T> GetValues() => _values.AsSpan(0, Nnz);
        public ReadOnlySpan<int> GetIndices() => _indices.AsSpan(0, Nnz);


        protected VBufferSparse((int[] indices, T[] values) valuesAndIndices, int length)
        {
            _values = new T[length];
            _indices = new int[length];
            Length = length;
            Nnz = valuesAndIndices.values.Length;
            Array.Copy(valuesAndIndices.values, _values, Nnz);
            Array.Copy(valuesAndIndices.indices, _indices, Nnz);
        }

        protected VBufferSparse(int[] indices, T[] values, int length)
        {
            _values = new T[length];
            _indices = new int[length];
            Length = length;
            Nnz = values.Length;
            Array.Copy(values, _values, Nnz);
            Array.Copy(indices, _indices, Nnz);
        }
    }
}
