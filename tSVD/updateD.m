function [D_tensor,D_tensor_pre] = updateD(D_tensor, J_tensor, E_tensor, Y1_tensor, Z_tensor, Y2_tensor, n1, n3, view_num, rho)
    D_tensor_pre = D_tensor;
    T1_tensor = J_tensor - E_tensor +Y1_tensor/rho;
    T2_tensor = Z_tensor + Y2_tensor/rho;
    hatJ = fft(J_tensor, [], 3);
    hatT1 = fft(T1_tensor, [], 3);
    hatT2 = fft(T2_tensor, [], 3);
    halfn3 = ceil((view_num+1)/2);
    for iv = 1:halfn3
        temp3 = hatJ(:,:,iv)'*hatJ(:,:,iv)+eye(n1,n1);
        temp4 = hatJ(:,:,iv)'*hatT1(:,:,iv)+hatT2(:,:,iv);
        hatD(:,:,iv) = pinv(temp3)*temp4;
    end
    for iv = halfn3+1 : view_num
        hatD(:,:,iv) = conj(hatD(:,:,n3+2-iv));
    end
    D_tensor = ifft(hatD, [], 3); 
end

