# EigenCore
[![.NET](https://github.com/modios/EigenCore/actions/workflows/dotnet.yml/badge.svg)](https://github.com/modios/EigenCore/actions/workflows/dotnet.yml)

EigenCore C# wrapper for the Eigen library.

## Usage 

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
VectorXD result = A.Solve(rhs) // DenseSolverType.ColPivHouseholderQR;

Console.WriteLine(result.ToString());
 
VectorXD, 3:
    
    -2 1 1
    
MatrixXD A = new MatrixXD("1  2  1; 2  1  0 ; -1  1  2");
VectorXD rhs = new VectorXD("3 3 4");
VectorXD result = A.Solve(rhs, DenseSolverType.LLT);  

Console.WriteLine(result.ToString());

VectorXD, 3:
    
    2.33 -1.67 4
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

  -0.97 0.143 0.196 
  -0.243 -0.571 -0.784 
   0 -0.809 0.588 

Console.WriteLine(result.R.ToString());

MatrixXD, 3 * 3:
  
   -4.12 -1.21 2.18 
    0 -1.24 0.285 
    0 0 0.392 

Console.WriteLine(result.P.ToString());

MatrixXD, 3 * 3:
 
    0 1 0 
    0 0 1 
    1 0 0 
```
## References
- https://eigen.tuxfamily.org/dox/group__QuickRefPage.html
- https://github.com/hughperkins/jeigen
- https://github.com/dotnet/machinelearning
