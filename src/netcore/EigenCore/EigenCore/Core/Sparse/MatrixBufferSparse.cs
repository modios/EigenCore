using System;

namespace EigenCore.Core.Sparse
{
    public class MatrixBufferSparse<T>
    {
        protected readonly T[] _values;
        protected readonly int[] _innerIndices;
        protected readonly int[] _outerStarts;

        public int Nnz { get; }
        public int Rows { get; }
        public int Cols { get; }

        public ReadOnlySpan<T> GetValues() => _values.AsSpan();

        public ReadOnlySpan<int> GetInnerIndices() => _innerIndices.AsSpan();

        public ReadOnlySpan<int> GetOuterStarts() => _outerStarts.AsSpan();

        public T GetValue(int index) => _values[index];

        protected MatrixBufferSparse(T[] values, int[] innerIndices, int[] outerStarts, int rows, int cols)
        {
            _values = values;
            _innerIndices = innerIndices;
            _outerStarts = outerStarts;
            Rows = rows;
            Cols = cols;
            Nnz = _values.Length;
        }
    }
}
