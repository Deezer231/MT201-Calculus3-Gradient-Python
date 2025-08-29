import sympy as sp

# Define variables
x, y = sp.symbols('x y')

# Example function
f = sp.sqrt(x**2 + y**2)

def compute_gradient(expr=f, point={x: sp.sqrt(17), y: 0}):
    """Compute gradient of a function at a given point."""
    grad = [sp.diff(expr, var) for var in (x, y)]
    grad_at_point = [g.subs(point) for g in grad]
    return grad, grad_at_point
