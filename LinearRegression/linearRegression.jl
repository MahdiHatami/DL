clearconsole()

using Knet, Plots

include(Knet.dir("data","housing.jl"))
x,y = housing()

predict(w,x) = w[1]*x .+ w[2]

loss(w,x,y) = mean(abs2, y - predict(w, x))

lossgradient = grad(loss)

function train(w, data; lr=.1)
    for (x,y) in data
        dw = lossgradient(w, x, y)
    	for i in 1:length(w)
    	    w[i] -= lr * dw[i]
    	end
    end
    return w
end

w = Any[ 0.1*randn(1,13), 0.0 ]

for i=1:10
    train(w, [(x,y)])
    println(loss(w,x,y))
end
display(plot(x', [y;predict(w,x)]', seriestype=:scatter, title="Data"))
