using Distributions, LinearAlgebra, DelimitedFiles
#Bayesian Neural Network

function network(topo, train, test, learn_rate)
    #initialize all fixed variable
    return topo, train, test, learn_rate
end

function weight_init(topo)
    weight_layer_1=rand(Normal(0, 1), (topo[1],topo[2]))./sqrt(topo[1]);
    #bias first layer
    bias_1=rand(Normal(0, 1), (1,topo[2]))./sqrt(topo[2]);
    #weight second layer
    weight_layer_2=rand(Normal(0, 1), (topo[2],topo[3]))./sqrt(topo[2]);
    #bias second layer
    bias_2=rand(Normal(0, 1), (1,topo[3]))./sqrt(topo[2]);
    #concatenate weights and biases
    weight_output=[weight_layer_1, bias_1, weight_layer_2, bias_2];
    return weight_output
end
# layer=weight_output(topo);

function sigmoid(x)
    return 1 ./ (1 .+ exp.(-x))
end

function compute_output(topo)
    #compute hidden and last layer output
    hidden_output=zeros(1,topo[2]);
    last_output=zeros(1,topo[3]);
    return last_output
end

function sampleEr(topo, actualout)
    error=compute_output(topo) .- actualout;
    sqerror=sum(error.^2)/topo[3];
    return sqerror
end

function forward_pass(X, topo, weight_vec)
    z_1=(reshape(X,(1,topo[1])))*weight_vec[1] .+ weight_vec[2];
    hid_out=sigmoid(z_1);
    z_2=hid_out*weight_vec[3] .+ weight_vec[4];
    last_out=sigmoid(z_2);
    return hid_out,last_out
end

    
function backward_pass(inp, desired, topo, weight_vec, learn_rate)
    hid_out, last_out=forward_pass(inp, topo, weight_vec);
    out_delta=((reshape(desired,(1,topo[3]))) .- last_out) .* (last_out .* (1 .- last_out)); 
    hid_delta=out_delta*transpose(weight_vec[3]) .* (hid_out .* (1 .- hid_out));
    h_o_layer=2 # hidden to output layer
    for i in 1:topo[h_o_layer]
        for j in 1:topo[h_o_layer+1]
            #update weight layer 2
            weight_vec[3][i,j] += learn_rate * out_delta[j] * hid_out[i]
        end
    end
    for k in 1:topo[h_o_layer+1]
        weight_vec[4][k] += -1 * learn_rate * out_delta[k]
    end
    
    i_h_layer=1 #input to hidden layer
    for i in 1:topo[i_h_layer]
        for j in 1:topo[i_h_layer+1]
            #update weight later 1
            weight_vec[1][i,j] += learn_rate * hid_delta[j] * inp[i] #placeholder for input
        end
    end
    for k in 1:topo[i_h_layer+1]
        weight_vec[2][k] += -1 * learn_rate * hid_delta[k]
    end
    return weight_vec
end

#decode a vector of proposal weights and biases to the correct weight vector

function decode(w_proposal, topo, weight_vec_prev)
    w_layer1size=topo[1] * topo[2];
    w_layer2size=topo[2] * topo[3];
    
    w_layer1=w_proposal[1:w_layer1size];
    weight_vec_prev[1]=reshape(weight_vec_prev[1],(topo[1],topo[2]));
    
    w_layer2=w_proposal[w_layer1size+1:w_layer1size + w_layer2size];
    weight_vec_prev[3]=reshape(weight_vec_prev[3],(topo[2],topo[3]));
    
    weight_vec_prev[2]=reshape(w_proposal[w_layer1size+w_layer1size+1:w_layer1size+w_layer1size+topo[2]], 1, topo[2]);
    weight_vec_prev[4]=reshape(w_proposal[w_layer1size+w_layer1size+topo[2]+1:w_layer1size+w_layer1size+topo[2]+topo[3]], 1, topo[3]);
    return weight_vec_prev
end

function encode(topo, weight_vec)
    w1=reshape(collect(Iterators.flatten(weight_vec[1])),(1,prod(size(weight_vec[1]))));
    w2=reshape(collect(Iterators.flatten(weight_vec[3])),(1,prod(size(weight_vec[3]))));
    #recollect to a vector
    weight_vec=hcat(w1,w2,weight_vec[2],weight_vec[4]);
    return weight_vec
end

function langevin_gradient(data, w_proposal, depth, topo, weight_vec, learn_rate) #BP with SGD
    #decode proposal parameter vector to corresponding weight vector 
    weight_vec=decode(w_proposal, topo, weight_vec);
    #how many n rows
    sz=size(data)[1];
    
    inp=zeros(1,topo[1]);
    desired=zeros(1,topo[3]);
    fx=zeros(sz);
    
    for i in 1:depth
        for i in 1:sz
            pat=i;
            inp=data[pat,1:topo[1]];
            desired=data[pat, (topo[1] + 1):(topo[1]+last(topo))];
            hid_out, last_out=forward_pass(inp, topo, weight_vec);
            weight_vec=backward_pass(inp, desired, topo, weight_vec, learn_rate);
        end
    end
    #encode weight and biases to vector again
    w_updated=encode(topo,weight_vec);
    
    return w_updated
end

function evaluate_proposal(data, w_proposal, topo, weight_vec)
    weight_vec=decode(w_proposal, topo, weight_vec);
    sz=size(data)[1];
    inp=zeros(1,topo[1]);
    desired=zeros(1,topo[3]);
    
    fx=zeros(sz,topo[3]);
    
    for i in 1:sz
        inp=data[i,1:topo[1]];
        hid_out, last_out=forward_pass(inp, topo, weight_vec);
        fx[i,:]=last_out;
    end
    return fx
end

function rmse(pred,actual)
    return sqrt(mean(((pred-actual).^2)))
end

function rmse_per_output(pred,actual,topo)
    individual_rmse=zeros(topo[3]);
    for i in 1:topo[3]
        individual_rmse[i]=sqrt(mean(((pred[1:size(pred)[1], i]-actual[1:size(actual)[1], i]).^2)))
    end
    return individual_rmse
end

function mae(pred, actual)
    return mean(abs.(pred-actual));
end
    
function mape(pred, actual)
    return 100 * mean(abs.((pred-actual)./(actual.+0.000001)));
end

    
function likelihood_func(data, w, tau_sq,topo, adapt_temp)
    y=data[:, topo[1]+1:topo[1]+last(topo)];
    fx=evaluate_proposal(data,w, topo, weight_vec);
    indi_rmse=rmse_per_output(fx,y,topo);
    root_mse=mean(indi_rmse);
    mean_abs_err=mae(fx,y);
    mean_abs_per_err=mape(fx,y);

    n=(size(y)[1]*size(y)[2]);
    p_1=(-n/2)*log(2*pi*tau_sq);
    p_2=0.5*tau_sq;

    log_lhood=p_1 - (p_2*sum((y-fx).^2));
    return [log_lhood/adapt_temp, fx, root_mse, indi_rmse, mean_abs_err, mean_abs_per_err];
end

function prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_sq,topo)
    h=topo[2];
    d=topo[1];
    part_1= -1 * ((d*h+h+2)/2) * log(sigma_squared);
    part_2= 1/ (2*sigma_squared) * (sum(w.^2));
    log_loss= part_1-part_2 - (1+nu_1)*log(tau_sq)-(nu_2/tau_sq);
    return log_loss
end
