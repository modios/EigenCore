using EigenCore.Core.Shared;
using System;
using System.Collections.Generic;
using System.Linq;

namespace EigenCore.Core.Sparse
{
    public class SparseVectorD : VectorSparseBase<double>
    {
        private bool IsEqual(SparseVectorD other)
        {
            if (Length != other.Length)
            {
                return false;
            }

            return ArrayHelpers.ArraysEqual(_values, other._values)
                && ArrayHelpers.ArraysEqual(_indices, other._indices);
        }

        public override bool Equals(object value)
        {
            if (ReferenceEquals(null, value))
            {
                return false;
            }

            if (ReferenceEquals(this, value))
            {
                return true;
            }

            if (value.GetType() != GetType())
            {
                return false;
            }

            return IsEqual((SparseVectorD)value);
        }
        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        public static SparseVectorD Random(int size, double percentageNonZeros, double min = 0, double max = 1, int seed = 0)
        {

            double maxMinusMin = max - min;
            int nnz = (int)Math.Floor(size * percentageNonZeros);
            (int, double)[] elements = new (int, double)[nnz];
            HashSet<int> visitedPosition = new HashSet<int>();
            if (_random == null) SetRandomState(seed);

            var count = 0;
            while (visitedPosition.Count < nnz)
            {
                int position = _random.Next(0, size);
                if (visitedPosition.Add(position))
                {
                    elements[count] =(position, maxMinusMin * _random.NextDouble() + min);
                    count += 1;
                }
            }

            return new SparseVectorD(elements, size);
        }

        public double Max() => _values.AsParallel().Max();

        public double Min() => _values.AsParallel().Min();

        public double Sum() => _values.AsParallel().Sum();

        public double Prod() => _values.AsParallel().Aggregate((product, nextElement) => product * nextElement);

        public double Mean() => _values.AsParallel().Average();

        public double Norm() => Math.Sqrt(_values.AsParallel().Sum(x => x * x));

        public double SquaredNorm() => _values.AsParallel().Sum(x => x * x);

        public double Lp1Norm() => _values.AsParallel().Sum(x => Math.Abs(x));

        public double LpInfNorm() => _values.AsParallel().Max(x => Math.Abs(x));

        public double Dot(SparseVectorD other)
        {
            return ArrayHelpers.ArraysDot(_values, other._values);
        }

        public SparseVectorD Add(SparseVectorD other)
        {
            return new SparseVectorD(_indices, ArrayHelpers.ArraysAdd(_values, other._values), Length);
        }

        public SparseVectorD Minus(SparseVectorD other)
        {
            return new SparseVectorD(_indices, ArrayHelpers.ArraysMinus(_values, other._values), Length); 
        }

        public void ScaleInplace(double scalar)
        {
            ArrayHelpers.ArraysScaleInplace(_values, scalar);
        }

        public SparseVectorD Scale(double scalar)
        {
            return new SparseVectorD(_indices, ArrayHelpers.ArraysScale(_values, scalar), Length);
        }

        internal SparseVectorD(int[] indices, double[] values, int length)
            : base(indices, values, length)
        {
        }

        public SparseVectorD((int indices, double values)[] valuesAndIndices, int length)
            : base(valuesAndIndices.SortByIndices(), length)
        {
        }
    }
}
