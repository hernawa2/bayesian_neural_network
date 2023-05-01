for i in 2:samples
    lx=rand(Uniform(0, 1));
    #using langevin gradient
    if lx<l_prob
        w_gd=langevin_gradient(train, w, depth, topo, weight_vec, learn_rate);
        w_proposal=rand.(Normal.(w_gd, step_w));
        w_prop_gd=langevin_gradient(train, w_proposal, depth, topo, weight_vec, learn_rate);
        wc_delta=(w-w_prop_gd);
        wp_delta=(w_proposal-w_gd);
        #compute variance
        sigma_sq=step_w^2;

        first=-0.5 * sum(wc_delta.^2)/sigma_sq;
        second= -0.5 * sum(wp_delta.^2)/sigma_sq;

        scaling_factor=0.25;

        diff_prop=first-second;
        diff_prop=diff_prop*scaling_factor/adapt_temp;
        langevin_count = langevin_count + 1
    else
        diff_prop=0;
        w_proposal=rand.(Normal.(w, step_w));
    end
    eta_pro=eta + rand(Normal(0, step_eta), 1)[1];
    tau_pro=exp(eta_pro);
    likelihood_proposal,pred_train,rmsetrain,indi_rmsetrain,maetrain,mapetrain=likelihood_func(train, w_proposal, tau_pro, topo, adapt_temp);
    _,pred_test,rmsetest,indi_rmsetest,maetest,mapetest=likelihood_func(test, w_proposal, tau_pro, topo, adapt_temp);
    prior_prop=prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal, tau_pro, topo);
    diff_prior=prior_prop-prior_current;
    diff_likelihood=likelihood_proposal-likelihood;
    surg_likeh_list[i,1]=likelihood_proposal*adapt_temp;
    mh_prob=min(1, exp(diff_likelihood+diff_prior+diff_prop));
    accept_list[i]=num_accepted;
    u=rand(Uniform(0, 1));
    prop_list[i,1:w_size]=w_proposal;
    likeh_list[i,1]=likelihood_proposal;
    if u<mh_prob
        num_accepted=num_accepted+1;
        likelihood=likelihood_proposal;
        prior_current=prior_prop;
        w=w_proposal;
        eta=eta_pro;
        acc_train[i]=0;
        acc_test[i]=0;

        pos_w[i,1:size(pos_w)[2]]=w_proposal;
        fxtrain_samples[i,:,:]=pred_train;
        fxtest_samples[i,:,:]=pred_test;
        rmse_train[i]=rmsetrain;
        rmse_test[i]=rmsetest;
        indi_rmse_train[i,1:size(indi_rmse_train)[2]]=indi_rmsetrain;
        indi_rmse_test[i,1:size(indi_rmse_test)[2]]=indi_rmsetest;
        mae_train[i]=maetrain;
        mae_test[i]=maetest;
        mape_train[i]=mapetrain;
        mape_test[i]=mapetest;
    else
        pos_w[i,1:size(pos_w)[2]]=pos_w[i-1,1:size(pos_w)[2]];
        fxtrain_samples[i,:,:]=fxtrain_samples[i-1,:,:];
        fxtest_samples[i,:,:]=fxtest_samples[i-1,:,:];
        rmse_train[i]=rmse_train[i-1];
        rmse_test[i]=rmse_test[i-1];
        indi_rmse_train[i,1:size(indi_rmse_train)[2]]=indi_rmse_train[i-1,1:size(indi_rmse_train)[2]];
        indi_rmse_test[i,1:size(indi_rmse_test)[2]]=indi_rmse_test[i,1:size(indi_rmse_test)[2]];
        mae_train[i]=mae_train[i-1];
        mae_test[i]=mae_test[i-1];
        mape_train[i]=mape_train[i-1];
        mape_test[i]=mape_test[i-1];
        acc_train[i]=acc_train[i-1];
        acc_test[i]=acc_test[i-1];
    end
    if mod(i,1000)==0
        println(i);
    end
end
