function x=mPRESS(xi, K)
L=length(xi);
if K>L
    disp('xi is shorter')
    return;
end
s=kron(xi,ones(1000,1));
sL=length(s);
z=sL/K;
zf=floor(z);
rem=sL-K*zf;
x=s(floor(rem/2)+1: zf: end-zf+1);



