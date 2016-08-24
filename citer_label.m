function s=citer_label(cit,rankl,k,bg_lt)
l=rankl(1:cit);
n=find(l==bg_lt);
if(n>0)
    s=1;
else
    s=0;
end
end

