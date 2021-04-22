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
        protected static int MaxRowsToPrint => 20;
        protected static int MaxColsToPrint => 20;

        protected static Random _random = default(Random);

        private static (int, int) GetRowsAndColsInfo(string valuesString)
        {
            string[] lines = valuesString.Split(";");
            int rows = lines.Length;
            string trimmedLine = Regex.Replace(lines[0], @"\s+", " ").Trim();
            string[] splitline = trimmedLine.Split(" ");
            int cols = splitline.Length;

            return (rows, cols);
        }

        private static T[] StringToFlatValues(string valuesString, Func<string, T> parser)
        {
            (int rows, int cols) = GetRowsAndColsInfo(valuesString);
            var inputValues = new T[rows * cols];

            string[] lines = valuesString.Split(";");
            int row = 0;

            foreach (string line in lines)
            {
                string lineTrim = Regex.Replace(line, @"\s+", " ").Trim();
                string[] splitline = lineTrim.Split(" ");

                if (splitline.Length != cols)
                {
                    throw new Exception("Unequal sized rows in " + valuesString);
                }

                for (int col = 0; col < cols; col++)
                {
                    inputValues[rows * col + row] = parser(splitline[col]);
                }

                row++;
            }

            return inputValues;
        }

        protected static (int, int) JaggedRowsAndColsInfo(T[][] inputValues) => (inputValues.Length, inputValues[0].Length);

        protected static (int, int) MultDimRowsAndColsInfo(T[,] inputValues) => (inputValues.GetLength(0), inputValues.GetLength(1));

        protected static T[] JaggedToFlatColumnWise(T[][] inputValues)
        {
            var numberOfRows = inputValues.Length;
            var numberOfCols = inputValues[0].Length;
            T[] values = new T[numberOfRows * numberOfCols];
            int index = 0;
            for (int col = 0; col < numberOfCols; col++)
            {
                for (int row = 0; row < numberOfRows; row++)
                {
                    values[index] = inputValues[row][col];
                    index += 1;
                }
            }

            return values;
        }

        protected static T[] MultyDimToFlatColumnWise(T[,] inputValues)
        {
            var numberOfRows = inputValues.GetLength(0);
            var numberOfCols = inputValues.GetLength(1);
            T[] values = new T[numberOfRows * numberOfCols];
            int index = 0;
            for (int col = 0; col < numberOfCols; col++)
            {
                for (int row = 0; row < numberOfRows; row++)
                {
                    values[index] = inputValues[row, col];
                    index += 1;
                }
            }

            return values;
        }

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

        public MatrixDenseBase(string valuesString, Func<string, T> parser)
            : base(() => StringToFlatValues(valuesString, parser))
        {
            (Rows, Cols) = GetRowsAndColsInfo(valuesString);
        }

#if DEBUG
        public string PrintFull()
        {
            StringBuilder matrix = new StringBuilder();
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Cols; j++)
                {
                    if (j != Cols - 1) matrix.Append(Get(i, j) + " ");
                    else matrix.Append(Get(i, j));
                }

                if (i == Rows - 1) return matrix.ToString();

                matrix.Append(";");
            }

            return default;
        }
#endif

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
