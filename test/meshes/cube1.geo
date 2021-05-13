SetFactory("OpenCASCADE");
Mesh.Algorithm = 1;   // Set 2D mesh algorithm as MeshAdapt
Box(1) = {-1, -1, -1, 2, 2, 2};
Physical Surface(1) = {3, 6, 1, 4, 5, 2};
