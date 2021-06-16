SetFactory("OpenCASCADE");
Mesh.Algorithm = 1;   // Set 2D mesh algorithm as MeshAdapt
Sphere(1) = {0, 0, 0, 2, -Pi/2, Pi/2, 2*Pi};
//+
Physical Surface(1) = {1};
