clc
clear all
close all
%% load 2-view data
load 2view.mat

%% test data of two views
%% Xs1 and Xt1 are the training and testing data for view 1
%% Xs2 and Xt2 are the training and testing data for view 2

Xt = [Xt1;Xt2]';
Yt = [Yt1;Yt2];

%% training data of two views
Xs = [Xs1;Xs2]';
Ys = [Ys1;Ys2];

%% sample size for each training view
ns1 = size(Xs1,1);
ns2 = size(Xs2,1);

%% build two graphs
%% Graph for Class Structure
options.WeightMode = 'Binary';
options.NeighborMode = 'Supervised';
options.gnd = Ys;
Wi = constructW(Xs',options);
Wi = full(Wi);

%% Graph for View structre
Wp = [ones(ns1), zeros(ns1,ns2);...
    zeros(ns2,ns1), ones(ns2)];
for i = 1:size(Wi,1)
    Wi(i,i)=1;
    for j = 1:size(Wi,2)
        if Wi(i,j)==1&&Wp(i,j)==1;
            Wp(i,j)=0;
        end
    end
end

%% calculate the laplacian graphs
Di = sum(Wi,2);
Di = diag(Di);
Li = (Di-Wi);

Dp = sum(Wp,2);
Dp= diag(Dp);
Lp = (Dp-Wp);

%% parameter for E term
lambda = 1e-2;

%% parameter for Graph regularizer
alpha =1e2;

%% reduced dimensionality
dp = 500;

%% call main function
%% Pa includes all the optimized projection in each iteration
Pa = RMSL(Xs, alpha, lambda, dp, Li, Lp);

%% classification
%% Choose the last iteration
%% You can choose the best one if there are valiation data
P = Pa{length(Pa)};

%% Exstract feature for test data
Zs = P'*Xs;
Zs = Zs*diag(sparse(1./sqrt(sum(Zs.^2))));
Zt = P'*Xt;
Zt = Zt*diag(sparse(1./sqrt(sum(Zt.^2))));
Cls = cvKnn(Zt, Zs, Ys, 1);
acc = length(find(Cls==Yt))/length(Yt);
fprintf('NN=%0.4f\n',acc);