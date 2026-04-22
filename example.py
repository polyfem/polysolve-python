import numpy as np
import scipy.sparse
import polysolve


class Quadratic(polysolve.Problem):
    def value(self, x):
        y = x - np.array([-2.0, 3.0, 1.0])
        return float(y @ y)

    def gradient(self, x):
        return 2.0 * (x - np.array([-2.0, 3.0, 1.0]))

    def hessian(self, x):
        return 2.0 * scipy.sparse.eye(x.size, format="csc")

    def post_step(self, iter_num, solver_info, x, grad):
        print(f"Iteration {iter_num}: x = {x}, grad = {grad}")

x, result = polysolve.minimize(
    Quadratic(),
    np.zeros(3),
    {
        "solver": "Newton",
        "line_search": {"method": "Backtracking"},
        "max_iterations": 100,
    },
    {"solver": "Eigen::SimplicialLDLT"},
)

print("Optimal point:", x)
print("Optimal value:", result)
