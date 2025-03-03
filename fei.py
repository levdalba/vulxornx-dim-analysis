import numpy as np

# Define the matrix A
A = np.array([[4, -1, 1], [-2, 3, -1], [2, 1, 5]])

# Define lambda values to test
lambda_values = [0, 2, 3, 4, 5]

for lam in lambda_values:
    # Compute determinant using NumPy
    A_minus_lambda_I = A - lam * np.eye(3)
    det_numpy = np.linalg.det(A_minus_lambda_I)

    # Compute the user's expression
    term1 = (4 - lam) * (3 - lam) * (5 - lam)
    term2 = -2 * (3 - lam)
    term3 = 4 - lam
    term4 = -2 * (5 - lam)
    det_expression = term1 + term2 + term3 + term4

    # Print results
    print(f"λ = {lam}:")
    print(f"  NumPy det(A - λI) = {det_numpy:.2f}")
    print(f"  User's expression = {det_expression:.2f}\n")
