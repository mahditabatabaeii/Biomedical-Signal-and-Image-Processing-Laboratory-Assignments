%% addpath
clc; close all;

functions_path = 'functions/';
results_path = 'results/';
data_path = 'data/';

addpath(functions_path)
addpath(results_path)
addpath(data_path)

%% Q1

% load image
img = imread(data_path+"/S2_Q1_utils/t2.jpg");
figure
imshow(img);
title('S2-Q1-utils/t2.jpg')
saveas(gcf,results_path + "fig.q1.0.png");

% first slice of the image
img_d = img(:,:,1);

% add gaussian noise to the image
varr = 0.015; m = 0;
img_d_noisy = imnoise(img_d,'gaussian',m,varr);

% binary kernel
[X,Y] = ndgrid(-size(img,1)/2:size(img,1)/2-1,-size(img,1)/2:size(img,1)-1);
k_size = 4;
kernel = zeros(size(img,1), size(img,2));
kernel(abs(X) < k_size & abs(Y) < k_size) = 1;
imshow(kernel)
title("Binary kernel, k size = " + k_size)
saveas(gcf,results_path + "fig.q1.00.png");

% normalize the kernel
kernel = kernel/sum(kernel(:));

% lets go to the fourier domain
kernel_fft = fftshift(fft2(kernel));
img_d_noisy_fft = fftshift(fft2(img_d_noisy));
G = kernel_fft.*img_d_noisy_fft;

% inverse fft
img_filtered = fftshift(abs(ifft2(G)));
img_filtered = img_filtered./max(img_filtered(:));

figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,2,1)
imshow(img_d)
title('Original Image - first slice')

subplot(2,2,2)
imshow(img_d_noisy)
title('Image with added gaussian noise')

subplot(2,2,3)
imshow(img_filtered)
title('Filtered using rect kernel')

% use imgaussfilt
img_filtered_2 = imgaussfilt(img_d,1);
subplot(2,2,4)
imshow(img_filtered_2)
title('Filtered using gaussian kernel on the original image')
saveas(gcf,results_path + "fig.q1.1.png");

%% Q2

% Load the image
img = imread(data_path+"/S2_Q2_utils/t2.jpg");
f = img(:,:,1);
figure;
imshow(f);
title('Original Image');
saveas(gcf, "results/fig.q2.0.png");

% Define Gaussian filter
f = double(image);
h = Gaussian(1, [256 256]);
g=conv2(f,h,'same');

% Fourier Transform
G_2=fftshift(fft2(ifftshift((g))));
H=fftshift(fft2(ifftshift((h))));
F=G_2./H;
f_recon=fftshift(ifft2(ifftshift(F)));

% Display results
figure('units', 'normalized', 'outerposition', [0 0 1 1]);
subplot(1, 3, 1);
imshow(f/max(f,[],'all'));
title('Original Image');
subplot(1, 3, 2);
imshow(g/max(g,[],'all'));
title('Blurred Image');
subplot(1, 3, 3);
imshow(f_recon/max(f_recon,[],'all'));
title('Reconstructed Image');
saveas(gcf, "results/fig.q2.1.png");

% Add Gaussian noise to blurred image
varr = 0.001;
g_noisy = g + randn(size(g)) * sqrt(varr); 
g_noisy = g_noisy / max(g_noisy(:));
G_noisy = fftshift(fft2(ifftshift(g_noisy)));

% Inverse filtering for noisy image reconstruction
F_noisy = G_noisy ./ H;
f_recon_noisy = abs(ifftshift(ifft2(ifftshift(F_noisy))));
f_recon_noisy = f_recon_noisy / max(f_recon_noisy(:));

% Display results
figure('units', 'normalized', 'outerposition', [0 0 1 1]);
subplot(1, 4, 1);
imshow(f/max(f,[],'all'));
title('Original Image');
subplot(1, 4, 2);
imshow(g/max(g,[],'all'));
title('Blurred Image');
subplot(1, 4, 3);
imshow(g_noisy);
title('Noisy Blurred Image');
subplot(1, 4, 4);
imshow(f_recon_noisy);
title('Reconstructed Noisy Image');
saveas(gcf, "results/fig.q2.2.png");

%% Q3
I = imread(data_path+"/S2_Q3_utils/t2.jpg");
I_d = I(:,:,1);
img = im2double(I_d);

% rescale image
scale = 1/4;
img = imresize(img, scale);

% K and h filter
K = zeros(size(img));
h = [0,1,0;...
    1,2,1;...
    0,1,0];
K(1:3,1:3) = h;

% create D by circulating K
[n, m] = size(img);
D = zeros(n*m, n*m);

count=1;
for c=1:64
    for r=1:64
        temp=circshift(K,[r-1 c-1]);
        D(count,:)=reshape(temp,1,64*64);
        count=count+1;
    end
end
spy(D)

img_vec = D*reshape(img,n*m,1);
out_img = reshape(img_vec,n,m);
img_vec_noisy = img_vec + 0.005*randn(length(img_vec),1);
img_recon = pinv(D)*img_vec_noisy;

figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,2,1);
imshow(K)
title('K image');

subplot(2,2,2);
imshow(out_img/max(out_img,[],'all'))
title('Df');

subplot(2,2,3);
imshow(img);
title('Orginial img')

subplot(2,2,4);
imshow(reshape(img_recon,m,n));
title('Reconstructed img')
saveas(gcf,results_path + "fig.q3.1.png");

%% Q4
img_vec = D*reshape(img,n*m,1);
g = img_vec + 0.005*randn(length(img_vec),1);

% gradient descent
beta = 0.01;
f = reshape(img,n*m,1);
f_learn = zeros(size(f));
errors = [];
error_st = 100000000000;
counter = 1;
th = 0.00001;

while error_st >= th
    f_learn_upd = f_learn + beta*D'*(g - D*f_learn);
    errors(counter) = immse(f,f_learn_upd)/norm(f);
    error_st = errors(counter);
    f_learn = f_learn_upd;
    counter = counter + 1;
end

figure;
scatter(1:counter-1,errors,25,'black','filled','black');
xlabel('Iteration'); ylabel('Recon MSE');
title('Reconstruction err in each iteration');
saveas(gcf,results_path + "fig.q4.0.png");

figure('units','normalized','outerposition',[0 0 1 1])
subplot(1,3,1);
imshow(img)
title('Orginal image');
subplot(1,3,2);
imshow(reshape(g,n,m)/max(reshape(g,n,m),[],'all'))
title('Noisy image');
subplot(1,3,3);
f_learn = f_learn/max(f_learn,[],'all');
imshow(reshape(f_learn,n,m))
title('Reconstructed image');

saveas(gcf,results_path + "fig.q4.1.png");

%% Functions 
function g = Gaussian(sigma, dims)

	rows = dims(1);
	cols = dims(2);
    slices = 1;
    D = 2;
    if length(dims)>2
        slices = dims(3);
        D = 3;
    end
    
	cr = ceil( (rows-1)/2 ) + 1;
	cc = ceil( (cols-1)/2 ) + 1;
    cs = ceil( (slices-1)/2) + 1;
    
    a = 1 / (2*sigma^2);
	g = zeros(rows,cols,slices);

    for s = 1:slices
        for c = 1:cols
            for r = 1:rows
                r_sh = r - cr;
                c_sh = c - cc;
                s_sh = s - cs;
                g(r,c,s) = exp( -a * (r_sh^2 + c_sh^2 + s_sh^2) );
            end
        end
    end
    
    g = g / (sqrt(2*pi)*sigma)^D;
    
end
