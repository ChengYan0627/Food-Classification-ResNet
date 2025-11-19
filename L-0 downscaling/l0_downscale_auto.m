function l0_downscale_auto(image_path, downscaling_factor, output_path, lambda_val, kappa_val)
%
%   image_path:        輸入圖像的完整路徑 (字串)。
%   downscaling_factor: 下採樣因子 (數值，例如 2 到 16)。
%   output_path:       輸出圖像的完整路徑 (字串)。
%   lambda_val:        L0 梯度最小化中的 lambda 參數 (數值)。
%   kappa_val:         L0 梯度最小化中的 kappa 參數 (數值)。

Im = im2double(imread(image_path));
[S, ~] = L0gDownscaling_sum(Im, (1/downscaling_factor), lambda_val, kappa_val);
imwrite(S, output_path);

end
