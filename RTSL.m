function [Z_tensor] = RTSL(X,truth,lambda,gamma,ind_folds)
    rho = 0.01; 
    for iv = 1:length(X)
        X1 = X{iv}';
        X1 = NormalizeFea(X1,1);
        ind_0 = find(ind_folds(:,iv) == 0);  % indexes of misssing instances
        X1(ind_0,:) = []; 
        data_e{iv} = X1';                           
        los_mark{iv} = ind_0;
    end
    numClust = length(unique(truth));
    view_num = length(X);
    data_num = length(truth);
    dim = zeros(1,view_num); 
    tq_l = cell(1,view_num);   
    kz_l = cell(1,view_num); 
    kz_e = cell(1,view_num); 
    tot_mark = 1:data_num;   
    data_l = cell(1,view_num); 
    data = cell(1,view_num);    
    sum_dim = 0;   
    for view_mark = 1:view_num    
        [dim(view_mark),~] = size(data_e{view_mark});
        sum_dim = sum_dim+dim(view_mark);
         los_mark{view_mark} = sort(los_mark{view_mark});  
         ext_mark = setdiff(tot_mark,los_mark{view_mark});
         tq_l{view_mark} = eye(data_num);
         tq_l{view_mark}(:,ext_mark) = [];
         l_num = length(los_mark{view_mark}); 
         e_num = data_num - l_num;  
         kz_l{view_mark} = zeros(l_num,data_num);
         kz_l{view_mark}(:,los_mark{view_mark}) = eye(l_num);
         kz_e{view_mark} = zeros(e_num,data_num);
         kz_e{view_mark}(:,ext_mark) = eye(e_num);
    end
    n1 = data_num;
    n2 = numClust;
    n3 = view_num;
    
    for iv = 1:view_num
        rand('seed',iv*100);
        U{iv} = abs(rand(size(X{iv},1),numClust));
        U{iv} = U{iv}/sum(sum(U{iv}));
        rand('seed',iv*100);
        P{iv} = abs(rand(numClust,size(X{iv},2)));
        P{iv} = P{iv}/sum(sum(P{iv}));
    end

    P_tensor = cat(3,P{:,:});
    J_tensor = zeros(n2,n1,n3);
    Y3_tensor = zeros(n2,n1,n3);
    
    D_tensor = zeros(n1,n1,n3);
    Z_tensor = D_tensor;
    Y2_tensor = D_tensor;
    
    E_tensor = zeros(n2,n1,n3);
    Y1_tensor = E_tensor;
    
    iter = 0;
    DEBUG = 1;
    max_rho = 1e+8;
    tol = 1e-8;
    max_iter = 100;
    
    I_tensor = zeros(n1,n1,n3);
    I_tensor(:,:,1) = eye(n1);
       
    while iter < max_iter
        iter = iter + 1;
        for iv = 1:view_num
            %% update X
            data_l{iv} = U{iv}*P{iv}*tq_l{iv}; 
            data{iv} = data_l{iv}*kz_l{iv} + data_e{iv}*kz_e{iv};        
            %% update U
            temp3 = data{iv}*P{iv}';
            ind2 = size(P{iv},1);
            temp4 = P{iv}*P{iv}'+gamma*eye(ind2);
            U{iv} = temp3*inv(temp4);           
            %% updade P
            J{iv} = J_tensor(:,:,iv); 
            Y3{iv} = Y3_tensor(:,:,iv);
            temp1 = U{iv}'*data{iv}+rho/2*(J{iv}-Y3{iv}/rho);
            ind1 = size(U{iv},2);
            temp2 = U{iv}'*U{iv}+rho/2*eye(ind1);
            P{iv} = inv(temp2)*temp1;
            P_tensor = cat(3,P{:,:}); 
        end
                
        %% update Zk
        Z_tensor_pre = Z_tensor;
        d = D_tensor(:);
        y2 = Y2_tensor(:);
        sX = [size(Z_tensor)];
        [z] = wshrinkObj(d - y2/rho,1/rho,sX,0,3);
        Z_tensor =reshape(z,sX);
        
        %% update Ek
        E_tensor_pre = E_tensor;
        R2=J_tensor-tprod(J_tensor,D_tensor)+Y1_tensor/rho;
        E_tensor = prox_l1( R2, lambda/rho );
        
        %% update Dk
        [D_tensor, D_tensor_pre] = updateD(D_tensor, J_tensor, E_tensor, Y1_tensor, Z_tensor, Y2_tensor, n1, n3, view_num, rho);

        %% update Jk
        [J_tensor,J_tensor_pre] = updateJ(D_tensor, J_tensor, E_tensor, Y1_tensor, I_tensor, P_tensor, Y3_tensor, view_num, n1, n3, rho);
              
        %% check convergence
        leq1 = J_tensor-tprod(J_tensor,D_tensor)-E_tensor;
        leq2 = Z_tensor-D_tensor;
        leq3 = P_tensor-J_tensor;
        leqm1 = max(abs(leq1(:)));
        leqm2 = max(abs(leq2(:)));
        leqm3 = max(abs(leq3(:)));
        difZ = max(abs(Z_tensor(:)-Z_tensor_pre(:)));
        difE = max(abs(E_tensor(:)-E_tensor_pre(:)));
        difD = max(abs(D_tensor(:)-D_tensor_pre(:)));
        difJ = max(abs(J_tensor(:)-J_tensor_pre(:)));
        err = max([leqm1,leqm2,leqm3,difZ,difE,difD,difJ]);
        if DEBUG && (iter==1 || mod(iter,20)==0)
            fprintf('iter = %d, err = %.8f, rho=%.2f\n', iter,err,rho);
        end
        if err < tol
            break;
        end            
        %% update Lagrange multiplier and  penalty parameter beta
        Y1_tensor = Y1_tensor + rho*leq1;
        Y2_tensor = Y2_tensor + rho*leq2;
        Y3_tensor = Y3_tensor + rho*leq3;
        rho = min(rho*1.2,max_rho);             
    end
end

