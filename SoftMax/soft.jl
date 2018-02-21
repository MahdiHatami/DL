using Knet

include(Knet.dir("data","mnist.jl"))
xtrn, ytrn, xtst, ytst = mnist()
dtrn = minibatch(xtrn, ytrn, 100)
dtst = minibatch(xtst, ytst, 100)

predict(w,x) = w[1]*mat(x) .+ w[2]

loss(w,x,ygold) = nll(predict(w,x), ygold)

lossgradient = grad(loss)

function train(w, data; lr=.1)
    for (x,y) in data
        dw = lossgradient(w, mat(x), y)
        for i in 1:length(w)
            w[i] -= lr * dw[i]
        end
    end
    return w
end

w = Any[ 0.1f0*randn(Float32,10,784), zeros(Float32,10,1) ]

println((:epoch, 0, :trn, accuracy(w,dtrn,predict), :tst, accuracy(w,dtst,predict)))

for epoch=1:3
    train(w, dtrn; lr=0.5)
    println((:epoch, epoch, :trn, accuracy(w,dtrn,predict), :tst, accuracy(w,dtst,predict)))
end
