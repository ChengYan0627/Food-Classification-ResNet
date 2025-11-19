disp('You can directly contact the author ''junjie.liu.cs@gmail.com'' if you have any questions during using this program')
disp('------------------------------------------------------------------------------------------------------------')

str = input('Input the original image path (such ''F:\\input.jpg''): ');
scale = input('Input the downscaling factor (2 ~ 16): ');
ostr = input('Input the output path for the downscaled image (such ''F:\\output.jpg''): ');
lambda_gra = input('Input the value of lambda (2e-2 ~ 2e-4): ');
kappa = input('Input the value of kappa (2 ~ 8): ');
%lambda_gra = 2e-4;
%kappa = 2;
%% ========================================================= %%

Im = im2double(imread(str));
[S kernel] = L0gDownscaling_sum( Im, (1/scale), lambda_gra, kappa );
imwrite(S, ostr);
