function [positions, time] = tracker_1(video_path, img_files, pos, target_sz, ...
	padding, kernel, lambda, output_sigma_factor, interp_factor, cell_size, ...
	features, show_visualization)
%TRACKER Kernelized/Dual Correlation Filter (KCF/DCF) tracking.
%   This function implements the pipeline for tracking with the KCF (by
%   choosing a non-linear kernel) and DCF (by choosing a linear kernel).
%
%   It is meant to be called by the interface function RUN_TRACKER, which
%   sets up the parameters and loads the video information.
%
%   Parameters:
%     VIDEO_PATH is the location of the image files (must end with a slash
%      '/' or '\').
%     IMG_FILES is a cell array of image file names.
%     POS and TARGET_SZ are the initial position and size of the target
%      (both in format [rows, columns]).
%     PADDING is the additional tracked region, for context, relative to 
%      the target size.
%     KERNEL is a struct describing the kernel. The field TYPE must be one
%      of 'gaussian', 'polynomial' or 'linear'. The optional fields SIGMA,
%      POLY_A and POLY_B are the parameters for the Gaussian and Polynomial
%      kernels.
%     OUTPUT_SIGMA_FACTOR is the spatial bandwidth of the regression
%      target, relative to the target size.
%     INTERP_FACTOR is the adaptation rate of the tracker.
%     CELL_SIZE is the number of pixels per cell (must be 1 if using raw
%      pixels).
%     FEATURES is a struct describing the used features (see GET_FEATURES).
%     SHOW_VISUALIZATION will show an interactive video if set to true.
%
%   Outputs:
%    POSITIONS is an Nx2 matrix of target positions over time (in the
%     format [rows, columns]).
%    TIME is the tracker execution time, without video loading/rendering.
%
%   Joao F. Henriques, 2014


	%if the target is large, lower the resolution, we don't need that much
	%detail
	resize_image = (sqrt(prod(target_sz)) >= 100);  %diagonal size >= threshold
	if resize_image,
		pos = floor(pos / 2);
		target_sz = floor(target_sz / 2);
	end


	%window size, taking padding into account
	window_sz = floor(target_sz * (1 + padding));
	
% 	%we could choose a size that is a power of two, for better FFT
% 	%performance. in practice it is slower, due to the larger window size.
% 	window_sz = 2 .^ nextpow2(window_sz);

	
	%create regression labels, gaussian shaped, with a bandwidth
	%proportional to target size
	output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
    
    % regression labels 不需要fft. 均值在中心的高斯分布
    y = gaussian_shaped_labels_1(output_sigma, floor(window_sz / cell_size));
    

    
    % 向量化
    y_vec = y2vec(y);
    
    
    
	%yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));

	%store pre-computed cosine window
	cos_window = hann(size(y,1)) * hann(size(y,2))';	
	
    
    
	
	if show_visualization,  %create video interface
		update_visualization = show_video(img_files, video_path, resize_image);
	end
	
	
	%note: variables ending with 'f' are in the Fourier domain.

	time = 0;  %to calculate FPS
	positions = zeros(numel(img_files), 2);  %to calculate precision

	for frame = 1:numel(img_files),
		%load image
		im = imread([video_path img_files{frame}]);
		if size(im,3) > 1,
			im = rgb2gray(im);
		end
		if resize_image,
			im = imresize(im, 0.5);
		end

		tic()

		if frame > 1,
			%obtain a subwindow for detection at the position from last
			%frame
			patch = get_subwindow(im, pos, window_sz);
            
            
			z = get_features(patch, features, cell_size, cos_window);
            z_sample = cirmat(z);
            
            response = z_sample * model_w;
            
            %根据response最大值确定最大响应位置
            index = find(response == max(response(:)), 1);
            
            horiz_delta = mod(index - 1, window_sz(2));
            vert_delta = fix((index - 1) / window_sz(2));
            
            
			
			%calculate response of the classifier at all shifts
% 			switch kernel.type
% 			case 'gaussian',
% 				kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
% 			case 'polynomial',
% 				kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);
% 			case 'linear',
% 				kzf = linear_correlation(zf, model_xf);
% 			end
%			response = real(ifft2(model_alphaf .* kzf));  %equation for fast detection

			%target location is at the maximum response. we must take into
			%account the fact that, if the target doesn't move, the peak
			%will appear at the top-left corner, not at the center (this is
			%discussed in the paper). the responses wrap around cyclically.
%			[vert_delta, horiz_delta] = find(response == max(response), 1);
			if vert_delta > window_sz(1) / 2,  %wrap around to negative half-space of vertical axis
				vert_delta = vert_delta - window_sz(1);
			end
			if horiz_delta > window_sz(2) / 2,  %same for horizontal axis
				horiz_delta = horiz_delta - window_sz(2);
			end
			pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];
		end

		%obtain a subwindow for training at newly estimated target position
		patch = get_subwindow(im, pos, window_sz);
        
        
        % 求解模型参数
        
% 		xf = fft2(get_features(patch, features, cell_size, cos_window));
% 
% 		%Kernel Ridge Regression, calculate alphas (in Fourier domain)
% 		switch kernel.type
% 		case 'gaussian',
% 			kf = gaussian_correlation(xf, xf, kernel.sigma);
% 		case 'polynomial',
% 			kf = polynomial_correlation(xf, xf, kernel.poly_a, kernel.poly_b);
% 		case 'linear',
% 			kf = linear_correlation(xf, xf);
% 		end
%		alphaf = yf ./ (kf + lambda);   %equation for fast training

        x = get_features(patch, features, cell_size, cos_window);
        
        x_sample = cirmat(x);
        

        w = (x_sample' * x_sample + lambda * eye(size(x_sample))) \ (x_sample') * y_vec;



		if frame == 1,  %first frame, train with a single image
			model_w = w;
			model_x_sample = x_sample;
		else
			%subsequent frames, interpolate model
			model_w = (1 - interp_factor) * model_w + interp_factor * w;
			model_x_sample = (1 - interp_factor) * model_x_sample + interp_factor * x_sample;
		end

		%save position and timing
		positions(frame,:) = pos;
		time = time + toc();

		%visualization
		if show_visualization,
			box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
			stop = update_visualization(frame, box);
			if stop, break, end  %user pressed Esc, stop early
			
			drawnow
% 			pause(0.05)  %uncomment to run slower
		end
		
	end

	if resize_image,
		positions = positions * 2;
	end
end

