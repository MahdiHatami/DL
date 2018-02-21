clearconsole()

using Knet, Plots

include(Knet.dir("data","housing.jl"))
x,y = housing()

predict(w,x) = w[1]*x .+ w[2]

loss(w,x,y) = mean(abs2, y - predict(w, x))

w = Any[ 0.1*randn(1,13), 0.0 ]

for i=1:15
    train(w, [(x,y)])
    println(loss(w,x,y))
end
# display(plot(x', [y;predict(w,x)]', seriestype=:scatter, title="Data"))
