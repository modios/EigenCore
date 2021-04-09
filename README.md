# EigenCore
[![.NET](https://github.com/modios/EigenCore/actions/workflows/dotnet.yml/badge.svg)](https://github.com/modios/EigenCore/actions/workflows/dotnet.yml)

EigenCore C# wrapper for the Eigen library.  
For now provides just a few methods and it's in very early stages.

## Usage 

### Matrix Constructors

```csharp

MatrixXD A = new MatrixXD("1 3 2; 0 2 1", 2, 3);

MatrixXD B = new MatrixXD(new double[][] { new double[] { 1, 3, 2 } , new double[] { 0, 2, 1 } });

MatrixXD C = new MatrixXD(new double[,] { { 1, 3, 2 }, { 0, 2, 1 } });

Console.WriteLine(A.ToString());

DenseMatrix, 2 * 3:
    
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

MatrixXD A = new MatrixXD("4 3; 3 2", 2, 2);
MatrixXD B = new MatrixXD("2 2; 1 1", 2, 2);
var result = A.Plus(B);
Console.WriteLine(result.ToString());

DenseMatrix, 2 * 2:
    
   6 5 
   4 3
```

```csharp

MatrixXD A = new MatrixXD("1 2; 3 5", 2, 2);
MatrixXD B = new MatrixXD("1 2; 3 2", 2, 2);
MatrixXD result = A.Mult(B);
Console.WriteLine(result.ToString());

DenseMatrix, 2 * 2  

    7 18 
    6 16 
```


```csharp

MatrixXD A = new MatrixXD("1 2 1; 2 5 2", 2, 3);
MatrixXD B = new MatrixXD("1 0 1; 1 1 0", 2, 3);
MatrixXD result = A.MultT(B);
Console.WriteLine(result.ToString());

DenseMatrix, 2 * 2  

    2 3 
    4 7 
```


## References
- https://eigen.tuxfamily.org/dox/group__QuickRefPage.html
- https://github.com/hughperkins/jeigen
