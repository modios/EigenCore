using System;

namespace EigenCore.Core.Dense
{
    public class VBufferDenseComplex<T>
    {
        protected readonly T[] _realValues;
        protected readonly T[] _imagValues;

        public readonly int Length;

        public ReadOnlySpan<T> GetRealValues() => _realValues.AsSpan(0, Length);

        public ReadOnlySpan<T> GetImagValues() => _imagValues.AsSpan(0, Length);

        public (T, T) GetItem(int index) => (_realValues[index], _imagValues[index]);

        public VBufferDenseComplex(T[] realValues, T[] complexValues)
        {
            Length = realValues.Length;
            _realValues = realValues;
            _imagValues = complexValues;
        }
    }
}
