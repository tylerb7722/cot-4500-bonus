# Tyler Boudreau
# 04/25/2023
# Bonus Assignment
# COT 4500

import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)


def gauss_seidel(A, b, x):
    x = x.astype(float)
    tolerance = 1e-6 # Set tolerence value
    error = tolerance+1
    iteration = 0 # Initialize iteration value
    max_iterations = 50 # Set max iterations value
    while(error>tolerance) and (iteration < max_iterations):  # Iterates through Gauss-Seidel Method  
        x0  = np.copy(x)
        for i in range(A.shape[0]): # Iterates through number of columns in matrix A
            left_numerator = (b[i] - np.dot(A[i,:i], x[:i])) # Gets left side numerator iteration value
            right_numerator = (np.dot(A[i,(i+1):], x0[(i+1):])) # Gets right side numerator iteration value
            denomenator = A[i ,i] # Gets left diagonal values
            x[i] = ((left_numerator-right_numerator)/denomenator)
        error = ((np.linalg.norm(x - x0))/(np.linalg.norm(x)))
        #print(error) # Enabling would show calculated error for each iteration
        iteration+=1 # Increase iteration each time error is calculated
        #print(iteration) # Enabling would show the iteration number each time
    print(iteration,"\n")
    
def jacobi(A,b,x):
    tolerence = 1e-6 # Set desired tolerence value
    iteration = 0 # Initialize iteration
    max_iterations = 50 # Set maxi iterations value
    error = tolerence+1
    x0 = np.ones_like(b, dtype=float)
    while (error>tolerence) and (iteration < max_iterations):
        # Jacobi Method:                                                                                                                                                                 
        D = np.diag(A)
        R = A - np.diagflat(D)                                                                                                                                                                         
        x = (b - np.dot(R,x)) / D
        error = np.linalg.norm(x - x0)
        #print(error) # Enabling would show calculated error for each iteration
        x0 = np.copy(x)
        iteration += 1 # Increase iteration each time error is calculated
        #print(iteration) # Enabling would show the iteration number each time
    print(iteration,"\n")

def function(value):
    return (value ** 3) - (value**2) + 2

def custom_derivative(value):
    return (3 * value*value) - (2 * value)

def newton_raphson(initial_approximation: float, tolerance: float, sequence: str):
    iteration_counter = 0
    # finds f
    x = initial_approximation
    f = eval(sequence)
    # finds f' 
    f_prime = custom_derivative(initial_approximation)

    approximation: float = f / f_prime
    while(abs(approximation) >= tolerance):
        # finds f
        x = initial_approximation
        f = eval(sequence)
        # finds f' 
        f_prime = custom_derivative(initial_approximation)
        # division operation
        approximation = f / f_prime
        # subtracts approximation from initial Aprox value and sets it equal
        initial_approximation -= approximation
        iteration_counter += 1
    print(iteration_counter,"\n")  

def apply_div_dif(matrix: np.array):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i+2):
            if (j >= len(matrix[i]) or matrix[i][j] != 0): # Check if value is already filled
                continue
            # Get left cell entry
            left: float = matrix[i][j-1]
            # Get diagonal left entry
            diagonal_left: float = matrix[i-1][j-1]
            numerator: float = left - diagonal_left # Calculate numerator
            # denominator is current i's x_val minus the starting i's x_val
            denominator = matrix[i][0] - matrix[i-2][0]
            operation = numerator / denominator # Perform division operation
            matrix[i][j] = operation
    return matrix

def hermite_interpolation(x_points, y_points, slopes):
    num_of_points = len(x_points)
    # Matrix size changes because of "doubling" up info for hermite
    matrix = np.zeros((num_of_points*2,num_of_points*2))
    # Fill with x values 
    for i in range(num_of_points):
        matrix[i*2][0] = x_points[i]
        matrix[i*2+1][0] = x_points[i]
    # Fill with y values 
    for i in range(num_of_points):
        matrix[i*2][1] = y_points[i]
        matrix[i*2+1][1] = y_points[i]
    # Fill with derivative values
    for i in range(num_of_points):
        matrix[i*2+1][2] = slopes[i]
    filled_matrix = apply_div_dif(matrix) # Apply divided difference method 
    print(filled_matrix,"\n")

def Runge_Kutta(f, t, y, h, n):
	for i in range(n):
		func_value = f(t, y) # Sets func_value equal to function
		t += h # Adds h value to t   
		next_value = float(f(t, y + h*func_value))
		y = float(y + h/2*(func_value + next_value))
	print(f'{y:.5f}'"\n") # Prints result to 5 decimal places


def main():
    # Initialize matrices used to solve for Problems 1 and 2
    A = np.array([[3,1,1],[1,4,1],[2,3,7]])
    b = np.array([1,3,0])
    guess = np.array([0,0,0])

    # Problem #1 Gauss-Seidel method
    gauss_seidel(A,b,guess)

    # Problem #2 Jacobi method
    jacobi(A,b,guess)

    # Problem #3 Newton Raphson method
    initial_approximation: float = .5 # Initial guess value 
    tolerance = 1e-6 # Error tolerence
    sequence: str = "x**3 - x**2 + 2" # Function as string
    newton_raphson(initial_approximation, tolerance, sequence)

    # Probelem #4 Divided difference method for Hermite olynomial aproximation matrix
    x_points = [0, 1, 2]
    y_points = [1, 2, 4]
    slopes = [1.06, 1.23, 1.55]
    hermite_interpolation(x_points, y_points, slopes)

    # Problem #5 Modified Eulers method or Runge-Kutta 2nd order:
    f = lambda t, y: (y-(t**3)) # Assign f to function with variable t
    t0 = 0 # Initial x or t point
    y0 = .5 # Initial y point
    fx = 3 # Value to estimate
    n = 100 # Number of Iterations
    h = (fx/100) # Value of h = 3/100
    Runge_Kutta(f,t0,y0,h,n)


if __name__ == "__main__":
    main()
