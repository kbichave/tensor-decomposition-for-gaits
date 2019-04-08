function xi=mSTRETCH(x, K)
L=length(x);
if K<L
    disp('x is longer')
    return;
end
a=K/L;
ac=ceil(a);
s=interp(x,ac);
xi=mPRESS(s, K);





