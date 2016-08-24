function distance = mindist(bag_A, bag_B)
num_A = size(bag_A.instance, 1);
num_B = size(bag_B.instance, 1);
for i = 1:num_A
    for j = 1:num_B
        inst_dist(i,j) = sum((bag_A.instance(i,:) - bag_B.instance(j,:)).^2);
    end
end
        dAB = min(inst_dist,[],2);
        dBA = min(inst_dist,[],1);
distance = (1/2)*(sum(dAB)+sum(dBA));
end