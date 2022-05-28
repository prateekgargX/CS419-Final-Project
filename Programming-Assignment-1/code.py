from ftplib import error_proto
from re import M, S
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

""" 
You are allowed to change the names of function "arguments" as per your convenience, 
but it should be meaningful.

E.g. y, y_train, y_test, output_var, target, output_label, ... are acceptable
but abc, a, b, etc. are not.
"""
def minmaxnorm(data_in):
    minimum = np.min(data_in)
    maximum = np.max(data_in)
    return (data_in - minimum)/(maximum-minimum)

def get_labeled_features(file_path):
    """Read data from train.csv and split into train and dev sets. Do any
       preprocessing/augmentation steps here and return final features.

    Args:
        file_path (str): path to train.csv

    Returns:
        phi_train, y_train, phi_dev, y_dev
    """
    data = pd.read_csv(file_path)
    data['type'] = data['type'].str.replace('white','0')
    data['type'] = data['type'].str.replace('red','1')
    data['type'] = pd.to_numeric(data['type'] )
    numpy_data = data.to_numpy()
    norm_np_data = np.apply_along_axis(minmaxnorm,0,numpy_data)
    train, dev = train_test_split(norm_np_data, test_size=0.3)
    phi_train=train[:, :-1]
    y_train=train[:,-1]
    phi_dev=dev[:,:-1]
    y_dev=dev[:,-1]
    phi_train_bias = np.hstack((phi_train,np.ones((phi_train.shape[0],1),dtype=phi_train.dtype)))
    phi_dev_bias = np.hstack((phi_dev,np.ones((phi_dev.shape[0],1),dtype=phi_dev.dtype)))
    n_train=phi_train.shape[0]
    n_dev=phi_dev.shape[0]
    return phi_train_bias, y_train.reshape((n_train,1)), phi_dev_bias, y_dev.reshape((n_dev,1))

def get_test_features(file_path):
    """Read test data, perform required preproccessing / augmentation
       and return final feature matrix.

    Args:
        file_path (str): path to test.csv

    Returns:
        phi_test: matrix of size (m,n) where m is number of test instances
                  and n is the dimension of the feature space.
    """
    data = pd.read_csv(file_path)
    data['type'] = data['type'].str.replace('white','0')
    data['type'] = data['type'].str.replace('red','1')
    data['type'] = pd.to_numeric(data['type'] )
    numpy_data = data.to_numpy()
    norm_np_data = np.apply_along_axis(minmaxnorm,0,numpy_data)
    bias_data = np.hstack((norm_np_data,np.ones((norm_np_data.shape[0],1),dtype=norm_np_data.dtype)))
    return bias_data

def compute_RMSE(phi, w , y) :
   """Return root mean squared error given features phi, and true labels y."""
   n=y.shape[0]
   err_vec=np.matmul(phi,w)-y
   error=np.sqrt(np.matmul(np.transpose(err_vec),err_vec)/n)
   return np.asscalar(error)

def RMSE_grad(phi, w , y) :
   """Return gradient of root mean squared error given features phi, and true labels y."""
   #print(phi)
   #print(y)
   n=phi.shape[0]
   #print(n)
   err_vec=np.matmul(phi,w)-y
   #print(err_vec)
   error=np.sqrt(np.matmul(np.transpose(err_vec),err_vec)/n)
   return np.matmul(np.transpose(phi),err_vec)/error/n

def generate_output(phi_test, w):
   """writes a file (output.csv) containing target variables in required format for Submission."""
   op=np.matmul(phi_test,w)
   np.savetxt("output.csv",op, delimiter=",")
   pass
    
   
def closed_soln(phi, y):
    """Function returns the solution w for Xw = y."""
    return np.linalg.pinv(phi).dot(y)
   
def gradient_descent(phi, y, phi_dev, y_dev) :
   # Implement gradient_descent using Mean Squared Error Loss
   # You may choose to use the dev set to determine point of convergence
    alpha=0.05 #Training_rate
    grad_norm_threshold=0.0001
    w=np.random.rand(phi.shape[1],1)
    d=RMSE_grad(phi,w,y)
    while abs(np.linalg.norm(d))>grad_norm_threshold:
        w=w-alpha*d
        d=RMSE_grad(phi,w,y)
    return w

def sgd(phi, y, phi_dev, y_dev) :
   # Implement stochastic gradient_descent using Mean Squared Error Loss
   # You may choose to use the dev set to determine point of convergence
    alpha=0.005 #Training_rate
    # grad_norm_threshold=0.0001
    num_steps=50000
    m=phi.shape[0]
    w=np.random.rand(phi.shape[1],1)
    d=RMSE_grad(phi[1].reshape(1,phi.shape[1]),w,y[1].reshape(1,1))
    #while abs(np.linalg.norm(d))>grad_norm_threshold:
    for i in range(num_steps):
        w=w-alpha*d
        randint_m=np.random.randint(m)
        phi_rand=phi[randint_m].reshape(1,phi.shape[1])
        d=RMSE_grad(phi_rand,w,y[randint_m].reshape(1,1))
    return w

def calc_p_norm(x,p) :
   # Implement gradient_descent with p-norm regularization using Mean Squared Error Loss
   # You may choose to use the dev set to determine point of convergence
    x=pow(x,p)
    x=np.sum(x)
    x=pow(x,1/p)
    return 

def pnorm(phi, y, phi_dev, y_dev, p) :
   # Implement gradient_descent with p-norm regularization using Mean Squared Error Loss
   # You may choose to use the dev set to determine point of convergence
    # Implement gradient_descent using Mean Squared Error Loss
   # You may choose to use the dev set to determine point of convergence
    alpha=0.05 #Training_rate
    grad_norm_threshold=0.0001
    lamb=0.1
    w=np.random.rand(phi.shape[1],1)
    d=RMSE_grad(phi,w,y)+lamb*p*pow(w,p-1)
    while np.linalg.norm(d)>grad_norm_threshold:
        w=w-alpha*d
        d=RMSE_grad(phi,w,y)+lamb*p*pow(w,p-1)
    return w   


def main():
    """ 
    The following steps will be run in sequence by the autograder.
    """
   ######## Task 2 #########
    phi, y, phi_dev, y_dev = get_labeled_features('train.csv')
    w1 = closed_soln(phi, y)
    w2 = gradient_descent(phi, y, phi_dev, y_dev)
    r1 = compute_RMSE(phi_dev, w1, y_dev)
    r2 = compute_RMSE(phi_dev, w2, y_dev)
    print(r1,r2)
    print('1a: ')
    print(abs(r1-r2))
    w3 = sgd(phi, y, phi_dev, y_dev)
    r3 = compute_RMSE(phi_dev, w3, y_dev)
    print('1c: ')
    print(abs(r2-r3))

    ######## Task 3 #########
    w_p2 = pnorm(phi, y, phi_dev, y_dev, 2)  
    w_p4 = pnorm(phi, y, phi_dev, y_dev, 4)  
    r_p2 = compute_RMSE(phi_dev, w_p2, y_dev)
    r_p4 = compute_RMSE(phi_dev, w_p4, y_dev)
    print('2: pnorm2')
    print(r_p2)
    print('2: pnorm4')
    print(r_p4)


    # print(w1)
    # print(w2)
    # print(w1)
    # print(w2)
    # print(w_p2)
    # print(w_p4)
    ######## Task 4 #########
    m=phi.shape[0]//100
    train_size=np.array([10,25,50,75,100])
    sizes=train_size*m
    err=[]
    for i in sizes:
        train_set=phi[0:i]
        train_var=y[0:i]
        w_i=closed_soln(train_set,train_var)
        r_i=compute_RMSE(phi_dev, w_i, y_dev)
        err.append(r_i)
    plt.plot(train_size,err)
    plt.xlabel("percentage of training data")
    plt.ylabel('RMSE on dev set')
    plt.title('Loss on development vs Training set size')
    plt.grid()
    plt.savefig('plot.png')
    plt.show()
    
    ######## Task 6 #########
    
    # Add code to run your selected method here
    # print RMSE on dev set with this method
    from sklearn.linear_model import SGDRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    reg = make_pipeline(StandardScaler(),SGDRegressor(max_iter=1000, tol=1e-3))
    reg.fit(phi, y.ravel())
    n=y_dev.shape[0]
    err_vec=reg.predict(phi_dev)-y_dev.ravel()
    error=np.sqrt(np.matmul(np.transpose(err_vec),err_vec)/n)
    print(error)
main()