SetFactory("OpenCASCADE");
Mesh.Algorithm = 1;   // Set 2D mesh algorithm as MeshAdapt
Box(1) = {-0.5, -0.5, -0.5, 1, 1, 1};
Physical Surface(1) = {3, 6, 1, 4, 5, 2};
