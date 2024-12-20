clear all
clc
addpath(genpath('dataset'));
addpath(genpath('tSVD'));
addpath(genpath('funs'));
Dataname = 'bbcsport4vbigRnSp';
percentDel = 0.3
f=1;
lambda = 1e4;
gamma = 1e-3;
Datafold = [Dataname,'_percentDel_',num2str(percentDel),'.mat'];
load(Dataname);
load(Datafold);
numClust = length(unique(truth));
view_num = length(X);
ind_folds = folds{f}; 
Z_tensor = RTSL(X,truth,lambda,gamma,ind_folds);

S = 0;
for k=1:view_num
    Z{k} = Z_tensor(:,:,k);
    S = S + abs(Z{k})+abs(Z{k}');
end

for iter_c = 1:10
    F = SpectralClustering(S./view_num, numClust);
    pre_labels = kmeans(F, numClust, 'maxiter', 1000, 'replicates', 20, 'emptyaction', 'singleton');
    result_LatLRR = ClusteringMeasure(truth, pre_labels);       
    AC(iter_c)    = result_LatLRR(1)*100;
    MIhat(iter_c) = result_LatLRR(2)*100;
    Purity(iter_c)= result_LatLRR(3)*100;
end
mean_ACC = mean(AC)
mean_NMI = mean(MIhat)
mean_PUR = mean(Purity)
