%Minimum Hausdroff Distance [UPDATED - 23 AUG 2016] [Modified]
%Minimum Hausdroff Distance [UPDATED - 31 AUG 2016] [Modified for including matrix manipulation]
function distance = hansf(bag_A, bag_B)
num_A = size(bag_A.instance, 1);
num_B = size(bag_B.instance, 1);
% for i = 1:num_A
%     for j = 1:num_B
%         inst_dist(i,j) = sum((bag_A.instance(i,:) - bag_B.instance(j,:)).^2);
%         if(inst_dist(i,j)==0)
%             inst_dist(i,j)=99999;%% This modification done to counter the dublicate instances.
%         end
%     end
% end
XX = sum(bag_A.instance.^2, 2);
CC = sum(bag_B.instance.^2, 2)';
XC = bag_A.instance * bag_B.instance';

inst_dist = sqrt(bsxfun(@plus, CC, bsxfun(@minus, XX, 2*XC)));%% This modification done to counter the duplicate instances.
inst_dist(inst_dist==0)=99999;

        dAB = min(inst_dist,[],2);
        dBA = min(inst_dist,[],1);
distance = max(min(dAB), min(dBA));
end