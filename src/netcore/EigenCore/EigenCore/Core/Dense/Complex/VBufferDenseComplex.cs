using System;

namespace EigenCore.Core.Dense
{
    public class VBufferDenseComplex<T>
    {
        protected readonly T[] _realValues;
        protected readonly T[] _complexValues;

        public readonly int Length;

        public ReadOnlySpan<T> GeRealValues() => _realValues.AsSpan(0, Length);

        public ReadOnlySpan<T> GeComplexValues() => _complexValues.AsSpan(0, Length);

        public (T,T) GetItem(int index) => (_realValues[index], _complexValues[index]);

        public VBufferDenseComplex(T[] realValues, T[] complexValues)
        {
            Length = realValues.Length;
            _complexValues = complexValues;
            _complexValues = complexValues;
        }
    }
}
