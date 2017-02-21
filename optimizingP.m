function Pn =optimizingP(P,X,Xn,Zn,E,Y1,alpha,mu,maxIterIn)
opts.record = 0;
opts.mxitr  =maxIterIn;
opts.xtol = 1e-3;
opts.gtol = 1e-3;
opts.ftol = 1e-4;
opts.tau = 1e-3;

Pn=OptStiefelGBB(P, @objHereInner, opts, X,Xn,Zn,E,Y1,alpha,mu);
end