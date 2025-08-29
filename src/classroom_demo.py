
"""
Gradient and Directional Derivative Demo
This script shows how to compute gradients and directional derivatives in Python using SymPy.
"""

import sympy as sp

print("# Q1: Gradient and directional derivative")

# Step 1: Define variables
x, y = sp.symbols('x y')
print("var('x y')  # Define variables")

# Step 2: Define the function
f = x**3 * sp.sin(y) + sp.cos(x) * y**2
print("f(x, y) = x^3*sin(y) + cos(x)*y^2  # Our example function")

# Step 3: Compute gradient components at (sqrt(17), 0)
point = {x: sp.sqrt(17), y: 0}
df_dx = sp.diff(f, x).subs(point)
df_dy = sp.diff(f, y).subs(point)
print("diff(f(x, y), x).subs((x==sqrt(17), y==0))  # Partial derivative wrt x at the point")
print("Out[1]:", df_dx)
print()
print("diff(f(x, y), y).subs((x==sqrt(17), y==0))  # Partial derivative wrt y at the point")
print("Out[2]:", df_dy)
print()
print(f"# Gradient is ({df_dx}, {df_dy})  # This is the gradient vector at our point")

# Step 4: Find the unit vector in the direction -i + 4j
vec = sp.Matrix([-1, 4])
vec_norm = vec.norm()
unit_vec = vec / vec_norm
print("vector([-1,4])/vector([-1,4]).norm()  # Unit vector in direction -i + 4j")
print("Out[3]:", tuple(unit_vec))
print()

# Step 5: Compute the directional derivative at (sqrt(17), 0) in direction -i + 4j
# This is the dot product of the gradient and the unit vector
directional_derivative = df_dx * unit_vec[0] + df_dy * unit_vec[1]
print("# directional derivative of f(x, y) at the point (sqrt(17), 0) in the direction -i+4j")
print("(0*-1/17*sqrt(17)) + (17*sqrt(17)*4/17*sqrt(17))  # Calculation")
print("Out[4]:", directional_derivative)
