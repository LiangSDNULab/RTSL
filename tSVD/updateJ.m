function [J_tensor,J_tensor_pre] = updateJ(D_tensor, J_tensor, E_tensor, Y1_tensor, I_tensor, P_tensor, Y3_tensor, view_num, n1, n3, rho)
        J_tensor_pre = J_tensor;
        M_tensor = I_tensor - D_tensor;
        W1_tensor = E_tensor - Y1_tensor/rho;
        W2_tensor = P_tensor + Y3_tensor/rho;
        hatM = fft(M_tensor, [], 3);
        hatW1 = fft(W1_tensor, [], 3);
        hatW2 = fft(W2_tensor, [], 3);
        halfn3 = ceil((view_num+1)/2);
        for iv = 1:halfn3
            temp1 = hatM(:,:,iv)*hatM(:,:,iv)'+eye(n1,n1);
            temp2 = hatW1(:,:,iv)*hatM(:,:,iv)'+hatW2(:,:,iv);
            hatJ(:,:,iv) = temp2*pinv(temp1);
        end
        for iv = halfn3+1 : view_num
            hatJ(:,:,iv) = conj(hatJ(:,:,n3+2-iv));
        end
        J_tensor = ifft(hatJ, [], 3);
end

