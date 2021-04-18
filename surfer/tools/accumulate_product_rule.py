def accumulate_product_rule(paramlist, gradients):
    grads = {}
    for params, grad in zip(paramlist, gradients):
        # all our gates only have one single parameter
        param = params[0]
        grads[param] = grads.get(param, 0) + grad

    return list(grads.values()), list(grads.keys())
