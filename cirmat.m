function x_sample = cirmat(A)

x_sample = zeros(numel(A));

% 生成循环矩阵
shift = size(A);

for i = 0:shift(1)-1  % 向下平移
    for j = 0:shift(2)-1 % 向右平移
        
        % 矩阵拉成向量
        x_sample(i*(shift(2))+j+1, :) = reshape(circshift(A, [i, j])', numel(A), 1);
    end
end

end

