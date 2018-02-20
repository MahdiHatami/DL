clearconsole()
using RDatasets
using Gadfly

include("logisticRegression.jl")
include("Evaluation.jl")

biopsy  = dataset("MASS","biopsy");
median_of_missing_V6 = median(biopsy[:V6][!isna(biopsy[:V6])])
biopsy[:V6][isna(biopsy[:V6])] = median_of_missing_V6
data = convert(Array{Float64,2}, biopsy[2:10]);
labels = convert(Array{Float64,1}, biopsy[:11] .== "benign");

params = Dict("alpha" => 0.001, "eps" =>  0.0005, "max_iter" => 10000.0)

learn_method = Dict("learning_method" => logistic_regression_learn, "params" =>  params, "inferencer" => predict);

predictions, eval_results = cross_validation(data, learn_method, data, labels, auc);

plot(
    x = eval_results[1][:,1],
    y= eval_results[1][:,2],
    Geom.line,
    Guide.xlabel("1 - Specificity"),
    Guide.ylabel("Sensivitiy")
    )
