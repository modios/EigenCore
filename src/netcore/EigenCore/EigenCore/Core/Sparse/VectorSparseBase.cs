using System;
using System.Linq;
using System.Text;

namespace EigenCore.Core.Sparse
{
    public class VectorSparseBase<T> : VBufferSparse<T>
    {
        protected static int MaxElements = 20;

        protected static Random _random = default(Random);

        public static void SetRandomState(int seed)
        {
            _random = new Random(seed);
        }

        public T Get(int index)
        {

            var position = Array.FindIndex(_indices, v => v == index);
            return position != -1 ? _values[position] : default;
        }

        public void Set(int index, T value)
        {
            var position = Array.FindLastIndex(_indices, v => v >= index);

            // if element exists replace value.
            if (_indices[position] == index)
            {
                _values[position] = value;
                return;
            }

            // if we insert new element, shift and place.
            Array.Copy(_values, position , _values, position + 1, Nnz - position);
            Array.Copy(_indices, position, _indices, position + 1, Nnz - position);
            _values[position] = value;
            _indices[position] = index;
            Nnz += 1;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="action"></param>
        /// <returns></returns>
        public int Count(Func<T, bool> action)
        {
            return _values.Count(v => action(v));
        }

        /// <summary>
        /// Replace values in array with the result of the action.
        /// The replace action is applied only to the
        /// "non-zero" elements for the sparse vector.
        /// </summary>
        /// <param name="action"></param>
        public void Replace(Func<T, T> action)
        {
            for (int i = 0; i < Nnz; i++)
            {
                _values[i] = action(_values[i]);
            }
        }

        public override string ToString()
        {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.Append(GetType().Name + ", " + Length + ":\n");
            stringBuilder.Append('\n');

            for (int i = 0; i < Length; i++)
            {
                if (i > MaxElements)
                {
                    stringBuilder.Append("...");
                    return stringBuilder.ToString().Trim();
                }

                stringBuilder.AppendFormat("{0:G3} ", Get(i));
            }

            return stringBuilder.ToString().Trim();
        }

        public VectorSparseBase((int[] indices, T[] values) valuesAndIndices, int length) : base(valuesAndIndices, length)
        {
        }

        public VectorSparseBase(int[] indices, T[] values, int length) : base(indices, values, length)
        {
        }
    }
}
