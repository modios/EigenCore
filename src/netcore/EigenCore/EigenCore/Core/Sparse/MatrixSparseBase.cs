using System;
using System.Text;

namespace EigenCore.Core.Sparse
{
    public class MatrixSparseBase<T> : MatrixBufferSparse<T>
    {
        protected static int MaxRowsToPrint => 20;
        protected static int MaxColsToPrint => 20;

        protected static Random _random = default(Random);

        public static void SetRandomState(int seed)
        {
            _random = new Random(seed);
        }

        public T Get(int row, int col)
        {
            int startColIndex = _outerStarts[col];
            int colElements = _outerStarts[col + 1] - _outerStarts[col];
            int endColIndex = startColIndex + colElements;

            for (int innerIndex = startColIndex; innerIndex < endColIndex; innerIndex++)
            {
                var columnIndex = _innerIndices[innerIndex];

                if (columnIndex == row)
                {
                    return _values[innerIndex];
                }

                if (columnIndex > row)
                {
                    return default(T);
                }
            }

            return default(T);
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


        public MatrixSparseBase((T[], int[], int[]) sparseInfo, int rows, int cols)
         : base(sparseInfo.Item1, sparseInfo.Item2, sparseInfo.Item3, rows, cols)
        {
        }

        public MatrixSparseBase(T[] values, int[] innerIndices, int[] outerStarts, int rows, int cols)
        : base(values, innerIndices, outerStarts, rows, cols)
        {
        }
    }
}
