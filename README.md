# EigenCore
[![.NET](https://github.com/modios/EigenCore/actions/workflows/dotnet.yml/badge.svg)](https://github.com/modios/EigenCore/actions/workflows/dotnet.yml)

EigenCore C# wrapper for the Eigen library.

## Usage 

### Matrix Operations

```csharp

MatrixXD A = new MatrixXD("1 2; 3 5", 2, 2);
MatrixXD B = new MatrixXD("1 2; 3 2", 2, 2);
MatrixXD result = A.Mult(B);
Console.WriteLine(result.ToString());

DenseMatrix, 2 * 2  

    7, 18, 
    6, 16 
```


```csharp

MatrixXD A = new MatrixXD("1 2 1; 2 5 2", 2, 3);
MatrixXD B = new MatrixXD("1 0 1; 1 1 0", 2, 3);
MatrixXD result = A.MultT(B);
Console.WriteLine(result.ToString());

DenseMatrix, 2 * 2  

    2, 3, 
    4, 7 
```

## References
- https://eigen.tuxfamily.org/dox/group__QuickRefPage.html
- https://github.com/hughperkins/jeigen
