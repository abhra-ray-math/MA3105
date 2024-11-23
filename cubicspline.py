import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
import pandas as pd


def jacobi(A, b, x0, tol, n_iterations=300):
    """
    Performs Jacobi iterations to solve the line system of
    equations, Ax=b, starting from an initial guess, ``x0``.
    
    Returns:
    x, the estimated solution
    """
    
    n = A.shape[0]
    x = x0.copy()
    x_prev = x0.copy()
    counter = 0
    x_diff = tol+1
    
    while (x_diff > tol) and (counter < n_iterations): #iteration level
        for i in range(0, n): #element wise level for x
            s = 0
            for j in range(0,n): #summation for i !=j
                if i != j:
                    s += A[i,j] * x_prev[j] 
            
            x[i] = (b[i] - s) / A[i,i]
        #update values
        counter += 1
        x_diff = (np.sum((x-x_prev)**2))**0.5 
        x_prev = x.copy() #use new x for next iteration
        
    
    print("Number of Iterations: ", counter)
    print("L2 Norm Error in Jacobi method: ", x_diff)
    return x


def cubic_spline(x, y, tol = 1e-10):
    """
    Interpolate using natural cubic splines.
    
    Generates a strictly diagonal dominant matrix then applies Jacobi's method,to get k_i parameters.
    """ 
    x = np.array(x)
    y = np.array(y)

    size = len(x)
    #print(size)
    delta_x = np.diff(x)
    #print(delta_x)
    delta_y = np.diff(y)
    
    ### Matrix A
    A = np.zeros(shape = (size,size))
    #print(A)
    b = np.zeros(size)
    #print(b)
    A[0,0]=2/delta_x[0]
    A[0,1]=1/delta_x[0]
    A[size-1,size-1]=2/delta_x[size-2]
    A[size-1,size-2]=1/delta_x[size-2]
    b[0]=3*(delta_y[0]/((delta_x[0])**2))
    b[size-1]=3*(delta_y[size-2]/((delta_x[size-2])**2))
    
    for i in range(1,size-1):
        A[i, i-1] = 1/delta_x[i-1]
        A[i, i+1] = 1/delta_x[i]
        A[i,i] = 2*((1/delta_x[i-1])+(1/delta_x[i]))
    ### Vector b
    for i in range(1,size-1):
        b[i] = 3*(delta_y[i]/((delta_x[i])**2) + delta_y[i-1]/((delta_x[i-1])**2))
        #print(b[i])
        
    ### Solves for c in Ac = b
    #print(A)
    #print(b)
    #print("Solution",np.linalg.solve(A,b))
    print('Jacobi Method Output:')
    c = jacobi(A, b, np.zeros(len(b)), tol = tol, n_iterations=1000)
    #print(c)
    return(c)
      

# Prepare the plot
plt.figure(figsize=(8, 6))


for n_ini in [5,10,20,40]:
    print("\n------------------------------------\n")
    print("Taking %d Points."%n_ini)
    #Declare points for interpolation
    x1=np.linspace(-1,1,n_ini)
    y1=np.exp(x1)


    #print(x,y)
    k=cubic_spline(x1,y1)
    k=list(k)
    x=list(x1)
    y=list(y1)
    a=list(np.zeros(len(k)-1))
    b=list(np.zeros(len(k)-1))
    #print(len(k))

    ### Use k_i parameters to generate a_i and b_i parameters.
    for i in range(len(k)-1):
        a[i] = (k[i]*(x[i+1]-x[i])) - (y[i+1]-y[i])
        b[i] = ((-k[i+1])*(x[i+1]-x[i])) + (y[i+1]-y[i])

    #print("\n",a,"\n",b)

    # Function to evaluate q(t) for a given interval
    def q_t(t, y_i, y_i1, a_i, b_i):
        return (1 - t) * y_i + t * y_i1 + t * (1 - t) * ((1 - t) * a_i + t * b_i)

    # Define the number of points for smooth plotting
    num_points = 1000

    # Loop through each interval to compute and plot q(t)
    all_y_vals=[]
    all_x_vals=[]
    for i in range(len(x) - 1):
        t_vals = np.linspace(0, 1, num_points)  # t in [0, 1]
        x_vals = np.linspace(x[i], x[i + 1], num_points)  # Map t to the actual x-interval pointwise
        y_vals = q_t(t_vals, y[i], y[i + 1], a[i], b[i])  # Evaluate q(t) for the interval
        all_x_vals+=list(x_vals)
        all_y_vals+=list(y_vals)
        # Plot the spline for this interval
        plt.plot(x_vals, y_vals)
        
    # Plot the original data points
    plt.scatter(x, y, label='Data Points (%d)'%n_ini,marker='.')

    #(USE ONLY IF FUNCTION KNOWN)
    #Find supnorm error in interpolation (DEPENDS ON INPUT FUNCTION)
    supnorm=0
    for i in range(len(all_x_vals)):
        if (abs((np.exp(all_x_vals[i])-all_y_vals[i]))>supnorm):
            supnorm=(abs(np.exp(all_x_vals[i])-all_y_vals[i]))
    print("Supnorm Error in Interpolation: ",(supnorm))
    hh=np.linspace(-1,1,10000)
    #plt.plot(hh,np.exp(hh))

# Finalize the plot
plt.title("Cubic Spline Interpolation")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
