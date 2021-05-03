using DataFrames
using Flux
using Mill

function sensitivity_nn_width(k_all, λ, nb_features)
	results = DataFrame(Algo = String[], k = Int[], l=Float64[], NLL_train = Float64[], NLL_test = Float64[])
	for k ∈ k_all
		# println(k)
		push!(results, get_results_mle(k,nb_features))
		push!(results, get_results_vadam(k,λ,nb_features))
	end
	return results
end


function get_results_mle(k,nb_features)

	model = BagModel(
	    ArrayModel(Dense(nb_features, k, Flux.tanh)),                      # model on the level of Flows
	    meanmax_aggregation(k),                                       # aggregation
	    ArrayModel(Chain(Dense(2*k+1, k, Flux.tanh), Dense(k, 2))))
	loss(x, y_oh) = Flux.logitcrossentropy(model(x).data, y_oh)

	#mle
	ps = Flux.params(model)
	Flux.train!(loss, ps, dta, opt)

	yt_pred = model(xt).data
	yt_pred = softmax(model(xt).data) #dle site a loss funkce
	yte_pred = model(xte).data
	yte_pred = softmax(model(xte).data)
	NLL_MLE_train = Flux.logitcrossentropy(model(xt).data,yt)
	NLL_MLE_test = Flux.logitcrossentropy(model(xte).data,yte)
	algo = "MLE"
	l = 0
	return(algo, k, l, NLL_MLE_train, NLL_MLE_test)

end

function get_results_vadam(k,λ,nb_features)

	model = BagModel(
	    ArrayModel(Dense(nb_features, k, Flux.tanh)),                      # model on the level of Flows
	    meanmax_aggregation(k),                                       # aggregation
	    ArrayModel(Chain(Dense(2*k+1, k, Flux.tanh), Dense(k, 2))))
	loss(x, y_oh) = Flux.logitcrossentropy(model(x).data, y_oh)

	#mle
	ps = Flux.params(model)
	(μ_vad,σ_vad) = Vadam.vadam!(loss,ps,dta,N_data,λ,10,1.0,0.01)
	for i ∈ 1:length(ps)
	    ps[i] .= μ_vad[i]
	end

	yt_pred = model(xt).data
	yt_pred = softmax(model(xt).data) #dle site a loss funkce
	yte_pred = model(xte).data
	yte_pred = softmax(model(xte).data)
	NLL_MLE_train = Flux.logitcrossentropy(model(xt).data,yt)
	NLL_MLE_test = Flux.logitcrossentropy(model(xte).data,yte)
	algo = "Vadam"
	return(algo, k, λ, NLL_MLE_train, NLL_MLE_test)

end
