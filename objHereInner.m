function [ff, ffP]=objHereInner(P,X,Xn,Zn,E,Y1,alpha,mu)

  %% calculate the objective function w.r.t. P
  G0 = P'*X*Zn*X'*P;
  G1 = P'*Xn-E+Y1/mu;
  ff=mu*sum(sum(G1.^2))+alpha*trace(G0);


  %% calculate the gradient function w.r.t. P
  P1 = 2*alpha*X*Zn*X'+mu*Xn*Xn';
  P2 = Xn*(E-Y1/mu)';
  ffP = P1*P-P2;
end
