function y_vec = y2vec(y)

y_vec = zeros(numel(y), 1);


% 生成循环矩阵
shift = size(y);
target_pos = fix(shift / 2);


for i = 0:shift(1)-1  % 向下平移
    for j = 0:shift(2)-1 % 向右平移
        
        
        if target_pos(1) + i <= shift(1)
            pos_y = target_pos(1) + i;
        else
            pos_y = target_pos(1) + i - shift(1);
        end
        
        if target_pos(2) + j <= shift(2)
            pos_x = target_pos(2) + j;
        else
            pos_x = target_pos(2) + j - shift(2);
        end
        
        % 矩阵拉成向量
        y_vec(i*(shift(2))+j+1) = y(pos_y, pos_x);
    end
end






end