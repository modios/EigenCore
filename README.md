# EigenCore
[![.NET](https://github.com/modios/EigenCore/actions/workflows/dotnet.yml/badge.svg)](https://github.com/modios/EigenCore/actions/workflows/dotnet.yml)

EigenCore C# wrapper of the Eigen library.

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

## References
- https://eigen.tuxfamily.org/dox/group__QuickRefPage.html
- https://github.com/hughperkins/jeigen
