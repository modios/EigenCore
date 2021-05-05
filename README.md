# EigenCore
[![.NET](https://github.com/modios/EigenCore/actions/workflows/dotnet.yml/badge.svg)](https://github.com/modios/EigenCore/actions/workflows/dotnet.yml)

C# wrapper for the Eigen library.

## Dense

### Vector Constructors

```csharp

VectorXD v = new VectorXD("1 2 5 6");

VectorXD v = new VectorXD( new double[] { 1, 2, 5, 6 });

Console.WriteLine(v.ToString());

VectorXD, 4:

    1 2 5 6

```

### Basic Vectors
```csharp

VectorXD zeros = VectorXD.Zeros(10);
VectorXD ones =  VectorXD.Ones(10);
VectorXD ident = VectorXD.Identity(10);
VectorXD random = VectorXD.Random(10);
VectorXD linspace = VectorXD.Linespace(1, 10, 10);
```

### Matrix Constructors

```csharp

MatrixXD A = new MatrixXD("1 3 2; 0 2 1");

MatrixXD B = new MatrixXD(new double[][] { new double[] { 1, 3, 2 } , new double[] { 0, 2, 1 } });

MatrixXD C = new MatrixXD(new double[,] { { 1, 3, 2 }, { 0, 2, 1 } });

Console.WriteLine(A.ToString());

MatrixXD, 2 * 3:
    
    1 3 2 
    0 2 1 
```

### Basic Matrices

```csharp

MatrixXD zeros = MatrixXD.Zeros(2, 3);
MatrixXD ones =  MatrixXD.Ones(2, 2);
MatrixXD ident = MatrixXD.Identity(3);
MatrixXD random = MatrixXD.Random(3,3);
MatrixXD diag = MatrixXD.Diag(new[] { 3.5, 2, 4.5 });

```


### Matrix Operations

```csharp

MatrixXD A = new MatrixXD("4 3; 3 2");
MatrixXD B = new MatrixXD("2 2; 1 1");
var result = A.Plus(B);

Console.WriteLine(result.ToString());

MatrixXD, 2 * 2:
    
   6 5 
   4 3
```

```csharp

MatrixXD A = new MatrixXD("1 2; 3 5");
MatrixXD B = new MatrixXD("1 2; 3 2");
MatrixXD result = A.Mult(B);

Console.WriteLine(result.ToString());

MatrixXD, 2 * 2:  

    7 18 
    6 16 
```

```csharp
MatrixXD A = new MatrixXD("1 2 4; 3 5 7");
MatrixXD B = A.Transpose();

Console.WriteLine(A.ToString());

MatrixXD, 2 * 3:
    
    1 2 4 
    3 5 7 
    
Console.WriteLine(B.ToString());

MatrixXD, 3 * 2:

    1 3 
    2 5 
    4 7 
```


```csharp
// X = A * B^T
MatrixXD A = new MatrixXD("1 2 1; 2 5 2");
MatrixXD B = new MatrixXD("1 0 1; 1 1 0");
MatrixXD result = A.MultT(B);

Console.WriteLine(result.ToString());

MatrixXD, 2 * 2:  

    2 3 
    4 7 
```

```csharp
// X = A + A^T
MatrixXD A = new MatrixXD("2 2; 1 1");
MatrixXD result = A.PlusT();

Console.WriteLine(result.ToString());

MatrixXD, 2 * 2:

    4 3
    3 2

```

```csharp
MatrixXD A = new MatrixXD("1  2  1; 2  1  0 ; -1  1  2");
double result = A.Determinant();
```           

```csharp
MatrixXD A = new MatrixXD("1  2  1; 2  1  0 ; -1  1  2");
MatrixXD result = A.Inverse();

Console.WriteLine(result.ToString());

MatrixXD, 3 * 3:

   -0.667 1 0.333 
    1.33 -1 -0.667 
   -1 1 1 
```     

### Matrix-Vector Operations

```csharp
MatrixXD A = new MatrixXD("1 -2 ; 2 5 ; 4 -2");
VectorXD v = new VectorXD(new[] { 3.0, -7.0 });
VectorXD result = A.Mult(v);

Console.WriteLine(result.ToString());

VectorXD, 3:  

   17 -29 26

```
### EigenSolvers

Computes eigenvalues and eigenvectors of general matrices.
[EigenSolver](https://eigen.tuxfamily.org/dox/classEigen_1_1EigenSolver.html)

```csharp

MatrixXD A = new MatrixXD("0 1; -2 -3");
EigenSolverResult result = A.Eigen();
VectorXD eigenvalues = result.Eigenvalues.Real();
MatrixXD eigenvectors = result.Eigenvectors.Real();

Console.WriteLine(eigenvalues.ToString());

VectorXD, 2:

  -1 -2
  
Console.WriteLine(eigenvectors.ToString());

MatrixXD, 2 * 2:
    
 0.707 -0.447 
-0.707  0.894 
 
```

Computes eigenvalues and eigenvectors of selfadjoint matrices.
[SelfAdjointEigenSolver](https://eigen.tuxfamily.org/dox/classEigen_1_1SelfAdjointEigenSolver.html)
```csharp

MatrixXD A = new MatrixXD("2 1; 1 2");
SAEigenSolverResult result = A.SymmetricEigen();
VectorXD eigenvalues = result.Eigenvalues;
MatrixXD eigenvectors = result.Eigenvectors;

Console.WriteLine(eigenvalues.ToString());

VectorXD, 2:

  1 3
  
Console.WriteLine(eigenvectors.ToString());

MatrixXD, 2 * 2:

 -0.707 0.707 
  0.707 0.707 
 
```

### SVD
```csharp
MatrixXD A = new MatrixXD("3 2 2 ; 2 3 -2");

SVDResult result = A.SVD(); // default Jacobi.
SVDResult result = A.SVD(SVDType.BdcSvd);

Console.WriteLine(result.U.ToString());

MatrixXD, 2 * 2:
    
  -0.707 0.707 
  -0.707 -0.707 
 
Console.WriteLine(result.S.ToString());

VectorXD, 2:

   5 3
   
Console.WriteLine(result.V.ToString());
 
MatrixXD, 3 * 2:
    
   -0.707 0.236 
   -0.707 -0.236 
   -2.22E-16 0.943 
```

### Least Squares
```csharp
MatrixXD A = new MatrixXD("-1 -0.0827; -0.737 0.0655; 0.511 -0.562 ");
VectorXD rhs = new VectorXD("-0.906 0.358 0.359");

// A^TAx = A^b
VectorXD result = A.LeastSquaresNE(rhs);

VectorXD result = A.LeastSquaresSVD(rhs); // default Jacobi.
VectorXD result = A.LeastSquaresSVD(rhs, SVDType.BdcSvd);

Console.WriteLine(result.ToString());
 
VectorXD, 2:
    
    0.463 0.0421
```

### Basic linear solving
```csharp
MatrixXD A = new MatrixXD("1 2 3; 4 5 6; 7 8 10");
VectorXD rhs = new VectorXD("3 3 4");
VectorXD result = A.Solve(rhs) // default DenseSolverType.ColPivHouseholderQR;

Console.WriteLine(result.ToString());
 
VectorXD, 3:
    
    -2 1 1
    
MatrixXD A = new MatrixXD("6 4 0;4 4 1;0 1 8");
VectorXD rhs = new VectorXD("3 3 4");
VectorXD result = A.Solve(rhs, DenseSolverType.LLT);  

Console.WriteLine(result.ToString());

VectorXD, 3:
    
    0.224 0.414 0.448
```

### QR decomposition
```csharp
MatrixXD A = new MatrixXD("1 -2 4; 1 -1 1;1 0 0");
QRResult result = A.QR(); // default QRType.HouseholderQR

Console.WriteLine(result.Q.ToString());

MatrixXD, 3 * 3:

   -0.577 0.707 0.408
   -0.577 0 -0.816 
   -0.577 -0.707 0.408

Console.WriteLine(result.R.ToString());

MatrixXD, 3 * 3:

  -1.73 1.73 -2.89 
   0 -1.41 2.83 
   0 0 0.816
```
```csharp
MatrixXD A = new MatrixXD("1 -2 4; 1 -1 1;1 0 0");
QRResult result = A.QR(QRType.ColPivHouseholderQR);

//Note: AP = QR

Console.WriteLine(result.Q.ToString());

MatrixXD, 3 * 3:

  -0.97   0.143 0.196 
  -0.243 -0.571 -0.784 
   0     -0.809 0.588 

Console.WriteLine(result.R.ToString());

MatrixXD, 3 * 3:
  
   -4.12 -1.21 2.18 
    0    -1.24 0.285 
    0     0    0.392 

Console.WriteLine(result.P.ToString());

MatrixXD, 3 * 3:
 
    0 1 0 
    0 0 1 
    1 0 0 
```
### LU Decomposition 

```csharp
// FullPivLU

MatrixXD A = new MatrixXD("-1 -0.562 -0.233;-0.737 -0.906 0.0388; 0.511 0.358 0.662; -0.0827 0.359 -0.931;  0.0655   0.869  -0.893");
FullPivLUResult result =  A.FullPivLU();

Console.WriteLine(result.L.ToString());

MatrixXD, 5 * 5:

 1       0       0    0 0 
 0.0827  1       0    0 0 
-0.0655  0.996   1    0 0 
 0.737  -0.231  -0.93 1 0 
-0.511  -0.596  0.729 0 1 

Console.WriteLine(result.U.ToString());

MatrixXD, 5 * 3:

-1 -0.233 -0.562 
 0 -0.912  0.405 
 0   0     0.428 
 0   0      0 
 0   0      0 

Console.WriteLine(result.P.ToString());

MatrixXD, 5 * 5:

1 0 0 0 0 
0 0 0 1 0 
0 0 0 0 1 
0 1 0 0 0 
0 0 1 0 0 

Console.WriteLine(result.Q.ToString());

MatrixXD, 3 * 3:

1 0 0 
0 0 1 
0 1 0 
```

### Concatenate and Slice Matrices
```csharp

// Provide indices to extract.
MatrixXD A = new MatrixXD("1 2 3; 4 5 5; 7 8 2");
MatrixXD result = A.Slice(new[] { 1, 2 }, new[] { 0, 1 });

Console.WriteLine(result.ToString());

MatrixXD, 2 * 2:
  
  4 5 
  7 8 

// Provide start and end indices.
MatrixXD A = new MatrixXD("1 2 3; 4 5 6; 7 8 2");
MatrixXD result = A.Slice(0, 1, 1, 2);

MatrixXD, 2 * 2:

  2 3 
  5 6 
  
// Concatenate Horizontal.
MatrixXD A = new MatrixXD("1 2; 4 5");
MatrixXD B = MatrixXD.Diag("1 0; 0 2");
MatrixXD result = A.Concat(B, ConcatType.Horizontal);

Console.WriteLine(result.ToString());

MatrixXD, 2 * 4:

  1 2 1 0 
  4 5 0 2 

// Concatenate vertical.
MatrixXD A = new MatrixXD("1 2; 4 5");
MatrixXD B = MatrixXD.Diag("1 0; 0 2");
MatrixXD result = A.Concat(B, ConcatType.Vertical);

Console.WriteLine(result.ToString());

MatrixXD, 4 * 2:

  1 2 
  4 5 
  1 0 
  0 2 
```

### Norms
```csharp

VectorXD v = new VectorXD("2 2 1");
v.Norm();
v.SquaredNorm();
v.Lp1Norm();
v.LpInfoNorm();

MatrixXD A = new MatrixXD("2 2 1; 1 2 -3; 1 0 1");
A.Norm();
A.SquaredNorm();
A.Lp1Norm();
A.LpInfNorm();

```

### Reductions
```csharp

A.Min() //  Min of all elements
A.Max() //  Max of all elements
A.Sum()) // Sum of all elements
A.Prod() // Product of all elements
A.Mean() // Mean of all elements.
A.Trace(); // Trace of the matrix.

VectorXD result = A.ColwiseMin(); // Min over columns
VectorXD result = A.RowwiseMin(); // Min over rows
VectorXD result = A.ColwiseMax(); // Max over columns
VectorXD result = A.RowwiseMax(); // Max over rows
VectorXD result = A.ColwiseSum(); // Sum over columns
VectorXD result = A.RowwiseSum(); // Sum over rows
VectorXD result = A.ColwiseProd(); // Product over columns
VectorXD result = A.RowwiseProd(); // Product over rows
VectorXD result = A.ColwiseMean(); // Mean over columns
VectorXD result = A.RowwiseMean(); // Mean over rows
```

### Count and Replace
```csharp

A.Count(x => double.IsNaN(x))); // count NaN values.
A.Count(x => double.IsInfinity(x))); // count infinity values.       

A.Replace(x => double.IsNaN(x) ? 0.0 : x); // replace NaN values with 0.0.
A.Replace(x => double.IsInfinity(x) ? 0.0 : x); // replace infinity values with 0.0. 
```
## Sparse

### Matrix Constructors

```csharp

(int, int, double)[] elements = {
                (0, 1, 3.0),
                (1, 0, 22),
                (2, 0, 7),
                (2, 1, 5),
                (4, 2, 14),
                (2, 3, 1),
                (1, 4, 17),
                (4, 4, 8)
            };

// Compressed sparse column (CSC) format.
SparseMatrixD A = new SparseMatrixD(elements, 5, 5);

Console.WriteLine(A.ToString());

SparseMatrixD, 5 * 5:

  0 3 0 0 0 
  22 0 0 0 17 
  7 5 0 1 0 
  0 0 0 0 0 
  0 0 14 0 8
  
```

### Direct Solvers
```csharp

SparseMatrixD A = new MatrixXD("6 4 0;4 4 1;0 1 8").ToSparse();
VectorXD rhs = new VectorXD("3 3 4");
VectorXD result = A.DirectSolve(rhs, DirectSolverType.SimplicialLLT);
VectorXD result = A.DirectSolve(rhs, DirectSolverType.SimplicialLDLT);
VectorXD result = A.DirectSolve(rhs, DirectSolverType.SparseLU);
VectorXD result = A.DirectSolve(rhs, DirectSolverType.SparseQR);

Console.WriteLine(result.ToString());

VectorXD, 3:

  0.224 0.414 0.448

```
### Iterative Solvers
```csharp

 (int, int, double)[] elements = {
                (0, 0, 6),
                (0, 1, 4),
                (0, 2, 0),
                (1, 0, 4),
                (1 ,1, 4),
                (1, 2, 1),
                (2, 0, 0),
                (2, 1, 1),
                (2, 2, 8)
            };
            
SparseMatrixD A = new SparseMatrixD(elements, 3, 3);
VectorXD rhs = new VectorXD("3 3 4");

VectorXD result = A.IterativeSolve(rhs); // Default IterativeSolverType.ConjugateGradient
VectorXD result = A.IterativeSolve(rhs, new IterativeSolverInfo(IterativeSolverType.BiCGSTAB));
VectorXD result = A.IterativeSolve(rhs, new IterativeSolverInfo(IterativeSolverType.GMRES));
VectorXD result = A.IterativeSolve(rhs, new IterativeSolverInfo(IterativeSolverType.GMRES, 2, 1e-2));
VectorXD result = A.IterativeSolve(rhs, new IterativeSolverInfo(IterativeSolverType.MINRES));
VectorXD result = A.IterativeSolve(rhs, new IterativeSolverInfo(IterativeSolverType.DGMRES))
VectorXD result = A.IterativeSolve(rhs, new IterativeSolverInfo(IterativeSolverType.LeastSquaresConjugateGradient));

i.e:
{EigenCore.Core.Sparse.LinearAlgebra.IterativeSolverResult}
    Error: 2.1912858061200212E-16
    Interations: 2
    Result: {VectorXD, 3: 0.224 0.414 0.448}
    Solver: ConjugateGradient
    Success: true
    
```

## References
- https://eigen.tuxfamily.org/dox/group__QuickRefPage.html
- https://github.com/hughperkins/jeigen
- https://github.com/dotnet/machinelearning
