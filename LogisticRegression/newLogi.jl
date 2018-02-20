clearconsole()
using Knet

function logit(y,x,w)
  b = x\y;  # use ols values as start
  maxit = 100;
  tol = 0.000001;

  #initialize breaks
  crit = 1.0;
  iter = 1;

  # loop
  while (iter < maxit) && (crit > tol)
    # Gradient & Hessian for logit
    (g,H) = LogitGradientAndHessian(y,x,b,w);

    # proposed change in coefficients
    db = -H\g

    # stepsize determination. Try steps of 1 or smaller
    s = 1.0
    L1 = LogitLogLikelihood(b+s*db,y,x,w)
    while s>tol
      s = s/2
      L2 = LogitLogLikelihood(b+s*db,y,x,w)
      if (L2-L1)<0 # log likelihood increases by less than 0/tol, abort
        s = 2*s # take previous step size, with step size the likelihood is declining
        break
      end
      L1 = L2; # reduce step-size, this is the new log likelihood
    end

    # take step
    b = b + s*db; # update coefficients
    crit = maximum(abs.(db)); # maximum absolute change in coefficient values (without step size like in original code...)
    iter = iter + 1;
  end # end of while

  return b,iter
end

function sigmoid(z::Float64)
  z = 1.0/(1.0+exp(-z))
  if z<0.0000001
    z = 0.0000001
  elseif z>0.99999999
    z = 0.99999999
  end
  return z
end

function LogitLogLikelihood(b,y,x,w)
  xb = x*b
  L = sum(w .* (y.*xb-log.(1+exp.(xb))))
  return L
end

function LogitGradientAndHessian(y,x,b,w)
  # gradient
  delta = sigmoid.(x*b)
  g = x'*(w.*(y.-delta))

  # pre-allocate Hessian
  k = size(x,2)
  H = zeros(eltype(g), (k,k))

  # compute Hessian (could also do only upper-right and copy to bottom-left)
  for n=1:size(x,1)
    tmp = w[n]*delta[n]*(1.0-delta[n])
    for kk1=1:k
      for kk2=1:k
        H[kk1,kk2] = H[kk1,kk2]-tmp*x[n,kk1]*x[n,kk2]
      end
    end
  end

  return g,H
end

include(Knet.dir("data","mnist.jl"))
x,y = mnist()

logit(y, x, 10)
