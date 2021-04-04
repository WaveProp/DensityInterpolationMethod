SetFactory("OpenCASCADE");
Sphere(1) = {0, 0, 0, 1, -Pi/2, Pi/2, 2*Pi};
//+
Box(2) = {-0.5, -0.5, 1.2, 1, 1, 1};
//+
Physical Surface(1) = {7, 3, 4, 6, 5, 2, 1};
