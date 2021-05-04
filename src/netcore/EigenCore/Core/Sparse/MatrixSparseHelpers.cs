using EigenCore.Core.Dense;
using System;
using System.Collections.Generic;
using System.Linq;

namespace EigenCore.Core.Sparse
{
    public static class MatrixSparseHelpers
    {
        const double ZeroTolerance = 10e-12;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="positionAndValues"></param>
        /// <param name="cols"></param>
        /// <returns></returns>
        public static (double[], int[], int[]) ToCCS(List<(int, int, double)> positionAndValues, int cols)
        {
            List<double> values = new List<double>();
            List<int> innerIndices = new List<int>();
            int[] outerStarts = new int[cols + 1];

            positionAndValues.Sort((x, y) => {
                var result = x.Item1.CompareTo(y.Item1);
                return result == 0 ? x.Item2.CompareTo(y.Item2) : result;
            });

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

        /// <summary>
        /// 
        /// </summary>
        /// <param name="positionAndValues"></param>
        /// <param name="cols"></param>
        /// <returns></returns>
        public static (double[], int[], int[]) ToCRS(IList<(int, int, double)> positionAndValues, int cols)
        {
            List<double> values = new List<double>();
            List<int> innerIndices = new List<int>();
            int[] outerStarts = new int[cols + 1];


            for (int row = 0; row < cols; row++)
            {
                var colElements = positionAndValues.Where(pV => pV.Item1 == row);
                int count = 0;
                foreach (var element in colElements)
                {
                    values.Add(element.Item3);
                    innerIndices.Add(element.Item2);
                    count++;
                }

                outerStarts[row + 1] = count + outerStarts[row];
            }

            return (values.ToArray(), innerIndices.ToArray(), outerStarts.ToArray());
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="sparseMatrix"></param>
        /// <returns></returns>
        public static MatrixXD ToDense(this SparseMatrixD sparseMatrix)
        {
            MatrixXD other = new MatrixXD(
                new double[sparseMatrix.Rows * sparseMatrix.Cols],
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

        /// <summary>
        ///
        /// </summary>
        /// <param name="denseMatrix"></param>
        /// <param name="tolerance"></param>
        /// <returns></returns>
        public static SparseMatrixD ToSparse(this MatrixXD denseMatrix, double tolerance = ZeroTolerance)
        {
            List<(int, int, double)> elements = new List<(int, int, double)>();

            for (int i = 0; i < denseMatrix.Rows; i++)
            {
                for (int j = 0; j < denseMatrix.Cols; j++)
                {
                    var value = denseMatrix.Get(i, j);
                    
                    if(Math.Abs(value) > tolerance)
                    {
                        elements.Add((i, j, denseMatrix.Get(i, j)));
                    }
                }
            }

            SparseMatrixD sparseMatrixD = new SparseMatrixD(elements, denseMatrix.Rows, denseMatrix.Cols);

            return sparseMatrixD;
        }
    }
}
