%Citation knn- Developed(max accuracy seen: 83.555%) 
%Note-here the output is the bag level accuracy.
%NOTE-Fubctions included here are: citer_label, habsf(Hausdorff Distance).
function [test_bl, test_il, test_bp, test_ip] = kNNc1(para, train_bags, test_bags)
global preprocess;
p = char(ParseParameter(para, {'-BagDistType';'-InstDistType';'-RefNum'; '-CiterRank'}, {'min';'euclidean';'5';'5'}));

bag_dist_type = p(1,:);
inst_dist_type = p(2,:);
ref = str2num(p(3,:));
citer = str2num(p(4,:));

if strcmp(bag_dist_type, 'max') && strcmp(bag_dist_type, 'min')
    error('Distance type must be either max or min in the kNN method for MIL!');    
end;

if strcmp(inst_dist_type, 'euclidean') && strcmp(inst_dist_type, 'cosine')
    error('Distance type must be either max or min in the kNN method for MIL!');    
end;

if isempty(train_bags)
    train_bags = MIL_Data_Load(preprocess.model_file);
elseif (~isempty(preprocess.model_file))    
    MIL_Data_Save(preprocess.model_file, train_bags);
end

ntestbags=length(test_bags);
ntrainbags=length(train_bags);

if ref > ntrainbags || ref < 1
    fprintf('num_ref  must be smaller than the number of training bags and positive');
    return;
end;

%predict the label for each testing bags
for i=1:ntestbags
    di_mat=0;
    for j=1:ntrainbags
        bg_dist(j)=hansf(test_bags(i),train_bags(j));
    end
    
    %creating a new training bag set with taking test bags one-by-one(one at a time) and
    %fitting them on the end of the set of training bags.  
    train_bagsnew=[train_bags test_bags(i)];
    ntrainbagsnew=length(train_bagsnew); %here ntrainbagsnew is the length value of new training set.
    
    %here in this section we are making new distance matrix which includes
    %the current test bag also along with all the train bags.
    for l=1:ntrainbagsnew
        for j=l+1:ntrainbagsnew
        di_mat(l,j)=hansf(train_bagsnew(l),train_bagsnew(j));
        di_mat(j,l)=di_mat(l,j);
        end
    di_mat(l,l)=9999999999;
    end
    
    %references
    testb_label_sample=0;
    [x y]=sort(bg_dist);
    for k=1:ref
        testb_label_sample(k)=train_bags(y(k)).label;
    end
    
    %citers
    id = ref + 1;
        for j=1:(ntrainbagsnew-1)
            [m, n] = sort(di_mat(j,:)); 
                p=citer_label(citer,n,j,ntrainbagsnew);
                %here in the above the citer_label sends whether we
                %have the current test bags in the distance matrix of new
                %set of train set or not. IF yes then p=1, IF no then p=0.
                if(p==1)
                testb_label_sample(id) = train_bags(j).label;
                id = id + 1;
                end
        end
        
    num_pos_label = sum(testb_label_sample == 1);
    num_neg_label = sum(testb_label_sample == 0);
    test_bl(i) = (num_pos_label >= num_neg_label);
    test_bp(i) = (num_pos_label / (num_pos_label + num_neg_label));
end
test_il = [];
test_ip = [];

end
