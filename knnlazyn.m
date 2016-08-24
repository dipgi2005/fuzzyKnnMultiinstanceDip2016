function [test_bag_label, test_inst_label, test_bag_prob, test_inst_prob] = knnlazyn(para, train_bags, test_bags)

global preprocess;
p = char(ParseParameter(para, {'-BagDistType';'-InstDistType';'-RefNum'; '-CiterRank'}, {'min';'euclidean';'5';'5'}));

bag_dist_type = p(1,:);
inst_dist_type = p(2,:);
num_ref = str2num(p(3,:));
rank_citer = str2num(p(4,:));

if strcmp(bag_dist_type, 'max') && strcmp(bag_dist_type, 'min')
    error('Distance type must be either max or min in the kNN method for MIL!');    
end;

if strcmp(inst_dist_type, 'euclidean') && strcmp(inst_dist_type, 'cosine')
    error('Distance type must be either max or min in the kNN method for MIL!');    
end;

%In the testing-only mode, the training file is missing, we get the model
%(in this lazy training case, just the training examples) from the model file
if isempty(train_bags)
    train_bags = MIL_Data_Load(preprocess.model_file);
elseif (~isempty(preprocess.model_file))    
    MIL_Data_Save(preprocess.model_file, train_bags);
end

num_train_bag = length(train_bags);
num_test_bag  = length(test_bags);

if num_ref > num_train_bag || num_ref < 1
    fprintf('num_ref  must be smaller than the number of training bags and positive');
    return;
end;

%predict the label for each testing bag
for i = 1:num_test_bag    
    bag_dist = zeros(num_train_bag, 1);
    
    select_label = [];
    for j = 1:num_train_bag
        bag_dist(j) = HausdorffDist(test_bags(i), train_bags(j), bag_dist_type, inst_dist_type);        
    end;
    [sort_dist, sort_idx] = sort(bag_dist);
    for j = 1 : num_ref %num_ref is neighbour
        select_label(j) = train_bags(sort_idx(j)).label;        
    end
    
    num_pos_label = sum(select_label == 1);
    num_neg_label = sum(select_label == 0);
    
    test_bag_label(i) = mode(select_label);
    test_bag_prob(i) = (num_pos_label / (num_pos_label + num_neg_label));
end;
ee=0;
ee=diff(num_test_bags);
test_inst_label = [];
test_inst_prob = [];

function err=diff(num_test_bags)
ercount=0;
for h=1:num_test_bag
%     for g=1:(size(test_bags(h).instance))
        if(select_label(h)~=test_bags(h).label)
            ercount=ercount+1;
        end
        err=(ercount/r)*100;
% test_inst_label = [];
% test_inst_prob = [];
end

function dist = HausdorffDist(bag_A, bag_B, type, metric)

num_A = size(bag_A.instance, 1);
num_B = size(bag_B.instance, 1);

%compute pair-wise distance
inst_dist = zeros(num_A, num_B);
for i = 1:num_A
    for j = 1:num_B
        inst_dist(i,j) = sum((bag_A.instance(i,:) - bag_B.instance(j,:)).^2);
    end        
end

dist_AB = min(inst_dist,[],2);
dist_BA = min(inst_dist,[],1);

if strcmp(type, 'max')
    dist = max(max(dist_AB), max(dist_BA));
else
    dist = max(min(dist_AB), min(dist_BA));
end