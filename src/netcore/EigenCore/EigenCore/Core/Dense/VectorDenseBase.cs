using System;
using System.Text;

namespace EigenCore.Core.Dense
{
    public class VectorDenseBase<T> : VBufferDense<T>
    {
        protected virtual int MaxElements => 20;

        protected static Random _random = default(Random);

        public static void SetRandomState(int seed)
        {
            _random = new Random(seed);
        }

        public T Get(int index) => _values[index];

        public T Set(int index, T value) => _values[index] = value;

        public override string ToString()
        {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.Append(GetType().Name + ", " + Length + ":\n");
            stringBuilder.Append('\n');

            for(int i =0; i<Length; i++)
            {
                if(i > MaxElements)
                {
                    stringBuilder.Append("...");
                    return stringBuilder.ToString().ToString().Trim();
                }

                stringBuilder.AppendFormat("{0:G3} ", _values[i]);
            }

            return stringBuilder.ToString().Trim();
        }


        public VectorDenseBase(T[] values) : base(values)
        {
        }
    }
}
