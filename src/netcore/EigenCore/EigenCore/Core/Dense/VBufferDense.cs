using System;

namespace EigenCore.Core.Dense
{
    public class VBufferDense<T>
    {
        protected readonly T[] _values;

        public readonly int Length;

        public ReadOnlySpan<T> GetValues() => _values.AsSpan(0, Length);

        public T GetItem(int index) => _values[index];

        public VBufferDense(T[] values)
        {
            Length = values.Length;
            _values = values;
        }
    }
}
