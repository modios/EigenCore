using System;

namespace EigenCore.Core.Dense
{
    public abstract class VBufferDense<T>
    {
        protected readonly T[] _values;

        /// <summary>
        /// Use only for reading raw data efficiently.
        /// </summary>
        public T[] Values => _values;

        public readonly int Length;

        public ReadOnlySpan<T> GetValues() => _values.AsSpan(0, Length);

        public T GetItem(int index) => _values[index];

        protected VBufferDense(T[] values)
        {
            _values = values;
            Length = _values.Length;
        }

        protected VBufferDense(Func<T[]> ActionToValues)
        {
            _values = ActionToValues();
            Length = _values.Length;
        }
    }
}
