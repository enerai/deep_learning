def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    """
    
    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
#     print(w.T.shape)
#     print(X.shape)
    A = sigmoid(np.dot(w.T,X)+b)                                  # compute activation
#     assert(A.shape[1]==X.shape[1])

#     print(type(np.log(A)))
#     print(np.log(A).shape)
#     print(Y.shape)
    cost = -np.sum(np.dot(Y,np.log(A).T)+np.dot(1-Y,np.log(1-A).T))/m                             # compute cost
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = np.dot(X,(A-Y).T)/m
    db = np.sum((A-Y))/m


    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
