using Knet, Plots, Images
using ASTInterpreter2

include(Knet.dir("data","mnist.jl"))

xtrn, ytrn, xtst, ytst = mnist()
dtrn = minibatch(xtrn, ytrn, 100)
dtst = minibatch(xtst, ytst, 100)

function predict(w, x)
    # mat is needed to convert the (28,28,1,N) x array to a (784,N) matrix so it can be used in matrix multiplication.
    return 1.0 ./ (1.0 + exp(-(w[1]*mat(x) + w[2])))
end

# nll computes the negative log likelihood of your predictions compared to the correct answers.
# loss(w,x,y) = nll(predict(w,x), y)
function loss(w,x,y)
    return mean(y .* log.(predict(w,x)) + (1 .- y) .* log.(1 .- predict(w,x)))
end

lossgradient = grad(loss)

function train(w, data, lr=.1)
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

for epoch=1:10
    train(w, dtrn; lr=0.5)
    println((:epoch, epoch, :trn, accuracy(w,dtrn,predict), :tst, accuracy(w,dtst,predict)))
end
