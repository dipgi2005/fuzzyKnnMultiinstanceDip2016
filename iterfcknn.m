%THIS CODE IS FOR MIL - FUZZY CITATION KNN - BAG-BASED & INSTANCE - BASED
%ALGORITHMS [23 AUGUST 2016]
%FINAL FUNCTION ITER(Number-of-iterations,'functon-based-name','dataset-name') -
%UPDATED ON 23-Aug 2016  [Function names -> 'knnfuzzy' or 'knnfinstnew']
function [finalmax,finalmean,finalstd,collect_output]=iterfcknn(n,func,dataset)
global preprocessor;
collect_output=[];
filename= strcat('Result_',dataset,'.text');
%filename=num2str(filename);
fid=fopen(filename,'w');
result=0;count=1;
% !For processing sparsr matrix we need to have -if 1 written after file name.
for mv=1.2:0.3:5
for refval=1:15
    citval1=refval; 
    citval5=(refval + 5);
for ctval=citval1:1:citval5    
for i=1:n
    ff=['classify -t ',dataset,' -sf 1 -mval ',num2str(mv),' -- cross_validate -t 10 -- ',func,' -RefNum ',num2str(refval),' -CiterRank ',num2str(ctval)];
    MIL_Run(ff);
    result(i)=ans.BagAccu;
end
finalmax=max(result)*100;
finalmean=mean(result)*100;
finalstd=std(result)*100;

collector=[n;count;mv;refval;ctval;finalmax;finalmean;finalstd];
collect_output=[collect_output,collector];

if(strcmp(func,'knnfinstnew'))
    fprintf(fid,'Function used- Fuzzy Citation knn Instance based\n');
else
    fprintf(fid,'Function used- Fuzzy Citation knn Bag based\n');
end
fprintf(fid,'Dataset used- %s \n',dataset);
fprintf(fid,'!!!! Run number - %d !!!!\n',count);
fprintf(fid,'Number of references- %d\n',refval);
fprintf(fid,'Number of citers- %d\n',ctval);
fprintf(fid,'Value of m- %d\n',mv);
fprintf(fid,'Max Value- %f\n Mean Value- %f\n Standard Deviation- %f\n\n\n',finalmax,finalmean,finalstd);

count=count+1;
% finalm=mean(result)*100;
end
end
end

% % for mv=1.1:0.2:6
% % for refval=1:20
% % for ctval=1:20    
% % for j=1:n
% %     MIL_Run('classify -t example.data -sf 1 -mval 4.9 -- cross_validate -t 10 -- knnfuzzy -RefNum 2 -CiterRank 4');
% %     result(j)=ans.BagAccu;
% % end
% % final2max=max(result)*100;
% % final2mean=mean(result)*100;
% % final2std=std(result)*100;
% % 
% % fprintf(fid,'Function used- Fuzzy Citation knn Bag based\n');
% % fprintf(fid,'Dataset used- example.data\n');
% % fprintf(fid,'Number of runs- %d\n',n);
% % fprintf(fid,'Number of references- %d\n',refval);
% % fprintf(fid,'Number of citers- %d\n',ctval);
% % fprintf(fid,'Value of m- %d\n',mv);
% % fprintf(fid,'Max Value- %f\n Mean Value- %f\n Standard Deviation- %f\n\n\n',final2max,final2mean,final2std);
% % % finalc=mean(result)*100;
% % end
% % end
% % end
fclose(fid);
end