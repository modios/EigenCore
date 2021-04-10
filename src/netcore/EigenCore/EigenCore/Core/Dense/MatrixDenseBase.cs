using System;
using System.Text;
using System.Text.RegularExpressions;

namespace EigenCore.Core.Dense
{
    /// <summary>
    /// The <see cref="MatrixDenseBase"/> class, of 
    /// column-major, dense format.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public abstract class MatrixDenseBase<T> : VBufferDense<T>
    {
        protected virtual int MaxRowsToPrint => 20;
        protected virtual int MaxColsToPrint => 20;

        protected static Random _random = default(Random);

        public static void SetRandomState(int seed)
        {
            _random = new Random(seed);
        }

        public int Rows { get; }

        public int Cols { get; }

        public abstract MatrixDenseBase<T> Clone();

        public void Set(int row, int col, T value)
        {
            _values[Rows * col + row] = value;
        }

        public void Set(int offset, T value)
        {
            _values[offset] = value;
        }

        public T Get(int row, int col)
        {
            return _values[Rows * col + row];
        }

        public T Get(int offset)
        {
            return _values[offset];
        }

        protected MatrixDenseBase(Func<T[]> ActionToValues, int rows, int cols)
                        : base(ActionToValues)
        {
            Rows = rows;
            Cols = cols;
        }

        protected MatrixDenseBase(T[] values, int rows, int cols)
           : base(values)
        {
            Rows = rows;
            Cols = cols;
        }

        public MatrixDenseBase(string valuesString, int rows, int cols, Func<string, T> parser)
            : base(new T[rows * cols])
        {
            string[] lines = valuesString.Split(";");
            Rows = rows;
            Cols = cols;
            int row = 0;

            foreach (string line in lines)
            {
                string lineTrim = Regex.Replace(line, @"\s+", " ").Trim();
                string[] splitline = lineTrim.Split(" ");

                if (splitline.Length != Cols)
                {
                    throw new Exception("Unequal sized rows in " + valuesString);
                }

                for (int col = 0; col < Cols; col++)
                {
                    Set(row, col, parser(splitline[col]));
                }

                row++;
            }
        }

        public override string ToString()
        {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.Append(GetType().Name + ", " + Rows + " * " + Cols + ":\n");
            stringBuilder.Append('\n');

            if (Rows * Cols <= MaxColsToPrint * MaxRowsToPrint)
            {
                for (int row = 0; row < Rows; row++)
                {
                    for (int col = 0; col < Cols; col++)
                    {
                        stringBuilder.AppendFormat("{0:G3} ", Get(row, col));
                    }
                    stringBuilder.Append('\n');
                }
            }
            else
            {
                int rows = (Rows > MaxRowsToPrint) ? MaxRowsToPrint : Rows;
                int cols = (Cols > MaxColsToPrint) ? MaxColsToPrint : Cols;
                for (int i = 0; i < rows; i++)
                {
                    int row = (Rows > MaxRowsToPrint && i > 3) ? Rows - i - 1 : i;
                    if (Rows > MaxRowsToPrint && i == 3)
                    {
                        stringBuilder.Append("...\n");
                    }
                    for (int j = 0; j < cols; j++)
                    {
                        int col = (Cols > MaxColsToPrint && j > 3) ? Cols - j - 1 : j;
                        if (Cols > MaxColsToPrint && j == 3)
                        {
                            stringBuilder.Append("... ");
                        }
                        else
                        {
                            stringBuilder.AppendFormat("{0:G3} ", Get(row, col));
                        }
                    }
                    stringBuilder.Append('\n');
                }
            }

            stringBuilder.Append('\n');
            return stringBuilder.ToString();
        }
    }
}
