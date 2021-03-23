# CUDA_GPU_Gaussian_Elimination

Implement Gaussian Elimination Algorithms on GPU to find the solution matrix X for the matrix equation AX = B.

The reference matrix A will be converted to Unit matrix and over different iterations, the pivot (diagonal axis of A) will be used to update the values in the according rows.
All elements below the pivot will be converted to 0 after the calculation. Thus, over the last iteration, matrix A becomes a triangle matrix and that is when the computation 
converges. The current values in X would be the solution. 

Since the algorithm can be computed independently for each row, we will pass each row to each thread of the GPU to speed up the calculations. More info on implementation details
and performance gain is reported in the report.pdf file.

