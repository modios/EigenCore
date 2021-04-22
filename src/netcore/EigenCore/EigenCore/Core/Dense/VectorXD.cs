using EigenCore.Eigen;
using System;
using System.Linq;

namespace EigenCore.Core.Dense
{
    public class VectorXD : VectorDenseBase<double>
    {
        private bool IsEqual(VectorXD other)
        {
            if (Length != other.Length)
            {
                return false;
            }

            return VectorHelpers.ArraysEqual(_values, other.GetValues().ToArray());
        }

        public static VectorXD Zeros(int size)
        {
            return new VectorXD(new double[size]);
        }

        public static VectorXD Ones(int size)
        {
            double[] input = Enumerable.Range(1, size).Select(n => 1.0).ToArray();
            return new VectorXD(input);
        }

        public static VectorXD Identity(int size, int position)
        {
            double[] input = new double[size];
            input[position] = 1.0;
            return new VectorXD(input);
        }

        public static VectorXD Random(int size, double min = 0, double max = 1, int seed = 0)
        {
            double[] input = new double[size];
            double maxMinusMin = max - min;

            if (_random == null) SetRandomState(seed);

            for (int i = 0; i < size; i++)
            {
                input[i] = maxMinusMin * _random.NextDouble() + min;
            }

            return new VectorXD(input);
        }

        public static VectorXD Linespace(double start, double end, int numberOfPoints)
        {
            double step = Math.Abs((end - start) / (numberOfPoints - 1.0));
            double[] input = new double[numberOfPoints];
            double value = start;
            for (int i = 0; i < numberOfPoints; i++)
            {
                input[i] = value;
                value += step;
            }

            return new VectorXD(input);
        }

        public double Max() => _values.AsParallel().Max();

        public double Min() => _values.AsParallel().Min();

        public double Sum() => _values.AsParallel().Sum();

        public double Prod() => _values.AsParallel().Aggregate((product, nextElement) => product * nextElement);

        public double Mean() => _values.AsParallel().Average();

        public double Dot(VectorXD other)
        {
            return EigenDenseUtilities.Dot(GetValues(), other.GetValues(), Length);
        }

        public double Norm()
        {
            return EigenDenseUtilities.Norm(GetValues(), Length);
        }

        public double SquaredNorm()
        {
            return EigenDenseUtilities.SquaredNorm(GetValues(), Length);
        }

        public double Lp1Norm()
        {
            return EigenDenseUtilities.Lp1Norm(GetValues(), Length);
        }

        public double LpInfoNorm()
        {
            return EigenDenseUtilities.LpInfNorm(GetValues(), Length);
        }

        public VectorXD Add(VectorXD other)
        {
            double[] outVector = new double[Length];
            EigenDenseUtilities.Add(GetValues(), other.GetValues(), Length, outVector);
            return new VectorXD(outVector);
        }

        public VectorXD Scale(double scalar)
        {
            double[] outVector = new double[Length];
            EigenDenseUtilities.Scale(GetValues(), scalar, Length, outVector);
            return new VectorXD(outVector);
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

            return IsEqual((VectorXD)value);
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        public VectorXD(string valuesString)
            : base(valuesString, (string value) => double.Parse(value))
        {
        }

        public VectorXD(double[] values) : base(values)
        {
        }
    }
}
