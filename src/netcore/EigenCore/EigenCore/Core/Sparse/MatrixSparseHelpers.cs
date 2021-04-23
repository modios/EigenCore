using EigenCore.Core.Dense;
using System.Collections.Generic;
using System.Linq;

namespace EigenCore.Core.Sparse
{
    public static class MatrixSparseHelpers
    {
        public static (double[], int[], int[]) ToCCS(IList<(int, int, double)> positionAndValues, int cols)
        {
            List<double> values = new List<double>();
            List<int> innerIndices = new List<int>();
            int[] outerStarts = new int[cols + 1];


            for (int col = 0; col < cols; col++)
            {
                var colElements = positionAndValues.Where(pV => pV.Item2 == col);
                int count = 0;
                foreach (var element in colElements)
                {
                    values.Add(element.Item3);
                    innerIndices.Add(element.Item1);
                    count++;
                }

                outerStarts[col + 1] = count + outerStarts[col];
            }

            return (values.ToArray(), innerIndices.ToArray(), outerStarts.ToArray());
        }


        public static MatrixXD ToDense(this SparseMatrixD sparseMatrix) 
        {
            MatrixXD other = new MatrixXD(
                new double[sparseMatrix.Rows* sparseMatrix.Cols],
                sparseMatrix.Rows,
                sparseMatrix.Cols);

            for (int i = 0; i < other.Rows; i++)
            {
                for (int j = 0; j < other.Cols; j++)
                {
                    other.Set(i, j, sparseMatrix.Get(i, j));
                }
            }

            return other;
        }
    }
}
