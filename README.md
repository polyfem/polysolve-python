# polysolve-python

Small Python binding for PolySolve's nonlinear solver interface.

```python
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


result = polysolve.minimize(
    Quadratic(),
    np.zeros(3),
    {
        "solver": "Newton",
        "line_search": {"method": "Backtracking"},
        "max_iterations": 100,
    },
    {"solver": "Eigen::SimplicialLDLT"},
)

print(result["x"])
print(result["status"])
print(result["info"])
```

Python subclasses must implement `value(x)`, `gradient(x)`, and `hessian(x)`. Optional PolySolve callbacks such as `solution_changed`,  `stop`, `is_step_valid`, and `max_step_size` can also be implemented on the subclass.
