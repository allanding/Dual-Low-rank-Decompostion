function Pa = RMSL(X, alpha, lambda, dp, Li, Lp)

%% min ||Zi||_*+||Zp||_*+\lambda||E||_1+\alpha G(Li,Lp,Zi,Zp), s.t., P'X = P'XZi+P'XZp+E.
%% for more detail, please check our AAAI-16 paper
%% @inproceedings{ding2016robust,
%%   title={Robust Multi-view Subspace Learning through Dual Low-rank Decompositions},
%%   author={Ding, Zhengming and Fu, Yun},
%%   booktitle={The Thirtieth AAAI Conference on Artificial Intelligence },
%%   year={2016}
%% }

%% initialize parameters
maxIter = 100;
[d, n] = size(X);

%% you could set these paramters manually
rho = 1.1;
max_mu = 1e6;
mu = 1e-1;


%% Initialize variables
Zi = zeros(n,n); %%class structure
Zp = zeros(n,n); %% view structure

E = zeros(dp,n); %% sparse error

Y1 = zeros(dp,n); %% laglange multiplier

rand('seed',1)
P = rand(d,dp); %% subspace projection

%% Start main loop
iter = 0;
warning off

%% to store all the projections
Pa = {};

while iter<maxIter
    iter = iter + 1;
    %% update Zi
    eta = norm(P'*X,2);
    Zip = Y1'*P'*X+mu*X'*P*(P'*X-P'*X*(Zi+Zp)-E);
    temp = Zi - 2*alpha*X'*P*P'*X*Zi*Li+Zip;
    [Ui,sigmai,Vi] = svd(temp,'econ');
    sigmai = diag(sigmai);
    svp = length(find(sigmai>1/(mu*eta)));
    if svp>=1
        sigmai = sigmai(1:svp)-1/(mu*eta);
    else
        svp = 1;
        sigmai = 0;
    end
    Zi = Ui(:,1:svp)*diag(sigmai)*Vi(:,1:svp)';
    
    
    %% update Zp
    temp = Zp + 2*alpha*X'*P*P'*X*Zp*Lp+Zip;
    [Up,sigmap,Vp] = svd(temp,'econ');
    sigmap = diag(sigmap);
    svp = length(find(sigmap>1/(mu*eta)));
    if svp>=1
        sigmap = sigmap(1:svp)-1/(mu*eta);
    else
        svp = 1;
        sigmap = 0;
    end
    Zp = Up(:,1:svp)*diag(sigmap)*Vp(:,1:svp)';
    
    
    %% update E
    Xn = X - X*(Zp+Zi);
    xmaz = P'*Xn;
    temp = xmaz+Y1/mu;
    E = max(0,temp - lambda/mu)+min(0,temp + lambda/mu);
    
    %% update P
    Zn = Zi*Li*Zi'-Zp*Lp*Zp';
    P1 = 2*alpha*X*Zn*X'+mu*Xn*Xn';
    P2 = Xn*(E-Y1/mu)';
    P = P1\P2;
    
    try
        P = orth(P);
        Pa{iter} = P;
        disp(['iter number is ' num2str(iter)])
    catch
        warning('P is optimized over.');
        break
    end
    
    %% update parameters
    leq1 = xmaz-E;
    Y1 = Y1 + mu*leq1;
    mu = min(max_mu,mu*rho);
    %% check convergence
    stopC = norm(leq1,'inf');
    disp(stopC)
    if stopC<10e-9
        break;
    end
end
