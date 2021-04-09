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
            _values = values;
            Length = _values.Length;
        }

        public VBufferDense(Func<T[]> ActionToValues)
        {       
            _values = ActionToValues();
            Length = _values.Length;
        }
    }
}
