% MIL is continuing...... Here we are generating a membership matrix along with
% corresponding classified-labels and original class. 
%Note-here the output is the bag level accuracy.
%NOTE-Functions included here are: citer_label, habsf(Hausdorff Distance).
function [test_bl, test_il, test_bp, test_ip] = knnfuzmult(para, train_bags, test_bags)
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

%compute the distances between any two training bags, when citers are used
% if rank_citer > 0
%     for i=1:num_train_bag
%         for j= i+1 : num_train_bag
%             dist_matrix(i,j) = HausdorffDist(train_bags(i),train_bags(j));            
%             dist_matrix(j,i) = dist_matrix(i,j);
%         end
%         dist_matrix(i,i) = intmax;
%     end    
% end
% --------------------------------------------------------------------------------------

% yp=input('Enter 1-Hanshroff Dist & 2-Min Dist & 3-Surjective Dist & 4-Avg Hausdroff Dist: ');
yp=4;
di_mat_tr=0;
    for l=1:ntrainbags        
        for j=l+1:ntrainbags
        di_mat_tr(l,j)=han(train_bags(l),train_bags(j),yp);
        di_mat_tr(j,l)=di_mat_tr(l,j);
        end
        di_mat_tr(l,l)=999999999;
    end
    u_tr=[];
for label=1:2
    for i=1:ntrainbags
        for j=1:ntrainbags
            bg_dist_tr(j)=di_mat_tr(i,j);
        end
    
    %creating a new training bag set with taking test bags one-by-one(one at a time) and
    %fitting them on the end of the set of training bags.  
%     train_bagsnew=[train_bags test_bags(i)];
%     ntrainbagsnew=length(train_bagsnew); %here ntrainbagsnew is the length value of new training set.
%     
    %here in this section we are making new distance matrix which includes
    %the current test bag also along with all the train bags.
    
    %references
    testb_label_sample_tr=0;
    [x y]=sort(bg_dist_tr);
    for k=1:ref
        testb_label_sample_tr(k)=train_bags(y(k)).label;
    end
    
    %citers
    id = ref + 1;
        for j=1:ntrainbags
            [m, n] = sort(di_mat_tr(j,:)); 
                p=citer_label(citer,n,j,i);
                %here in the above the citer_label sends whether we
                %have the current test bags in the distance matrix of new
                %set of train set or not. IF yes then p=1, IF no then p=0.
                if(p==1)
                testb_label_sample_tr(id) = train_bags(j).label;
                id = id + 1;
                end
        end

        
        if(train_bags(i).label==label-1)
            u_tr(i,label)=0.51+(size(find(testb_label_sample_tr==(label-1)),2)/(ref+citer))*0.49;
        else
            u_tr(i,label)=0.49*(size(find(testb_label_sample_tr==(label-1)),2)/(ref+citer));
        end
    end
end
% ------------------------------------------------------------------------------------

%predict the label for each testing bags
u_te=0;
for i=1:ntestbags
    di_mat=0;
    for j=1:ntrainbags
        bg_dist(j)=han(test_bags(i),train_bags(j),yp);
    end
    
    %creating a new training bag set with taking test bags one-by-one(one at a time) and
    %fitting them on the end of the set of training bags.  
    train_bagsnew=[train_bags test_bags(i)];
    ntrainbagsnew=length(train_bagsnew); %here ntrainbagsnew is the length value of new training set.
    
    %here in this section we are making new distance matrix which includes
    %the current test bag also along with all the train bags.
    for l=1:ntrainbagsnew
        for j=l+1:ntrainbagsnew
        di_mat(l,j)=han(train_bagsnew(l),train_bagsnew(j),yp);
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
    id = ref + 1;cm=1;cit=0;
        for j=1:(ntrainbagsnew-1)
            [m, n] = sort(di_mat(j,:)); 
                p=citer_label(citer,n,j,ntrainbagsnew);
                %here in the above the citer_label sends whether we
                %have the current test bags in the distance matrix of new
                %set of train set or not. IF yes then p=1, IF no then p=0.
                if(p==1)
                testb_label_sample(id) = train_bags(j).label;
                cit(cm)=j;
                cm=cm+1;
                id = id + 1;
                end
        end
   
         ur=0;um=0;m=4.9;
         for label=1:2
         for u=1:ref
             ur(u)=(1/di_mat(ntrainbagsnew,y(u)))^(2/(m-1));
             um(u)=u_tr(y(u),label);
         end
         
         un=u+1;cu=1;
         
         for ux=un:(ref+(cm-1))
             ur(ux)=(1/di_mat(ntrainbagsnew,n(cu)))^(2/(m-1));
             um(ux)=u_tr(cit(cu),label);
             cu=cu+1;
         end
         
         u_te(i,label)=sum(um.*ur)/sum(ur);
         end
%             u_te(:,:)
            [maxval,maxind]=max(u_te(i,:));
            test_bl(i)= maxind-1; 
            test_bp(i) = u_te(i,2)/sum(u_te(i,:));
            [u_te(i,:),(maxind-1),test_bags(i).label]
end
                
%     num_pos_label = sum(testb_label_sample == 1);
%     num_neg_label = sum(testb_label_sample == 0);
%     test_bl(i) = (num_pos_label >= num_neg_label);
%     test_bp(i) = (num_pos_label / (num_pos_label + num_neg_label));
    test_il = [];
    test_ip = [];
end