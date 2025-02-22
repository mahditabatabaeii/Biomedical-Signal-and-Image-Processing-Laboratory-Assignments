%% addpath
clc; close all;

results_path = 'results/';
data_path = 'data/';

addpath(results_path)
addpath(data_path)

%% Q1
% Read the image
img = imread(data_path + "/S1_Q1_utils/t1.jpg");

% Display the image
figure
imshow(img);
title('S1-Q1-utils/t1.jpg') 
saveas(gcf, results_path + "fig.q1.1.png"); 

% Select the first slice
img_d = img(:, :, 1);
img_d_slcted_row = img_d(128, :);

% Length of the selected row
N = length(img_d_slcted_row);

% DFT of the selected row
dfft_row = fft(img_d_slcted_row, N);

% fftshift to center the zero-frequency component
dfft_row = fftshift(dfft_row);


figure('units', 'normalized', 'outerposition', [0 0 1 1]);

% Plot the magnitude of the DFT
subplot(2, 1, 1); 
scale = linspace(-pi, pi, N); 
plot(scale, abs(dfft_row));
xlim([scale(1) scale(end)]); 
title('Magnitude of DFT of the row 128');
xlabel('Frequency');
grid on; 

% Plot the phase of the DFT
subplot(2, 1, 2);
plot(scale, angle(dfft_row)); 
xlim([scale(1) scale(end)]);
title('Phase of DFT of the row 128'); 
xlabel('Frequency');
grid on; 
saveas(gcf, results_path + "fig.q1.2.png"); 

% Convert the image slice to double for FFT2 
fft2_img = fft2(double(img_d));

figure('units', 'normalized', 'outerposition', [0 0 1 1]);

% Plot the original first slice of the image
subplot(1, 3, 1);
imshow(img_d);
title('First Slice of Image');

% fftshift and the log magnitude
subplot(1, 3, 2);
fft_rot_2 = log10(abs(fftshift(fft2_img)));
fft_rot_2 = fft_rot_2 / max(max(abs(fft_rot_2)));
imshow(fft_rot_2); 
title('2d FFT with fftshift'); 

% Log magnitude without fftshift
subplot(1, 3, 3);
fft_rot_2 = log10(abs(fft2_img)); 
fft_rot_2 = fft_rot_2 / max(max(abs(fft_rot_2)));
imshow(fft_rot_2); 
title('2d FFT without fftshift');

saveas(gcf, results_path + "fig.q1.3.png");

%% Q2 
% G matrix
L = 256;
[X,Y] = ndgrid(-L/2:L/2-1, -L/2:L/2-1);
G = zeros(L,L);
G(X.^2 + Y.^2 <= L-1) = 1;

% L matrix
F = zeros(L,L);
F(100,50) = 1;
F(120,48) = 2;

% Convolution of G and F using Fourier Transform
fft_G = fft2(G);
fft_F = fft2(F);
fft_out = fft_G.*fft_F;
conv_img = fftshift(ifft2((fft_out)));
conv_img = conv_img/max(max(conv_img));

% Plot G, F, and their convolution result
figure('units','normalized','outerposition',[0 0 1 1])
subplot(1,3,1)
imshow(G)
title('G Matrix')
subplot(1,3,2)
imshow(F)
title('L Matrix')
subplot(1,3,3)
imshow(conv_img)
title('Convolution of Images')

saveas(gcf,results_path + "fig.q2.1.png");

% pd image and conv with G
img = imread(data_path+"/S1_Q2_utils/pd.jpg");
img_d = double(img(:,:,1));

% Normalization
img_d = img_d/max(max(img_d));

% Convolution of the pd image and G using Fourier Transform
fft_img = fft2(img_d);
fft_out = fft_img.*fft_G;
conv_img = fftshift(ifft2(fft_out));
conv_img = 255*conv_img/max(max(abs(conv_img)));
conv_img = uint8(conv_img);

% Plot the original image and the convolution result
figure
subplot(1,2,1)
imshow(img_d)
title('S1-Q2-utils/pd.jpg')

subplot(1,2,2)
imshow(conv_img)
title('pd convolved with G')
saveas(gcf,results_path + "fig.q2.2.png");

%% Q3
% The input image
img = imread(data_path+"/S1_Q3_utils/ct.jpg");
[d1, d2, d3] = size(img);

% 2D FFT of the image and shift zero frequency to the center
fft_img = fftshift(fft2(img));
zoom_s = 2;
new_img = zeros(zoom_s*d1, zoom_s*d2, 3);

%offset for centering the original image in the larger matrix
ms = round(d1*(zoom_s/2 - 0.5));
ns = round(d2*(zoom_s/2 - 0.5));

% Place the frequency content in the center of the zero-padded matrix
new_img(ms:(d1+ms-1), ns:(d2+ns-1), :) = fft_img;

% Inverse FFT to return to the spatial domain
final = abs(ifft2(ifftshift(new_img)));
f_fin = final(ms:(d1+ms-1), ns:(d2+ns-1), :);

% Plot the original and zoomed images
figure('units','normalized','outerposition',[0 0 1 1])
subplot(1,2,1)
imshow(img,[])
title('Original Image');

subplot(1,2,2)
imshow(uint8(f_fin*zoom_s^2),[]);
title('Zoomed Image');
saveas(gcf,results_path + "fig.q3.1.png");

%% Q4-1

% Step 1: Read the image
image = imread('S1_Q4_utils/ct.jpg');
image_gray = rgb2gray(image); % Convert to grayscale if necessary

% Display the original image
figure;
imshow(image_gray);
title('Original Image');

% Step 2: Define the spatial shift
[m, n] = size(image_gray); % Get the size of the image
shift_x = 20; % Shift 20 units to the right
shift_y = 40; % Shift 40 units downward

% Create the Fourier shift kernel
[x, y] = meshgrid(0:(n-1), 0:(m-1));
shift_kernel = exp(-2j * pi * (shift_x * x / n + shift_y * y / m));

% Step 3: Compute the Fourier Transform of the image
image_fft = fft2(double(image_gray));

% Apply the shift in the Fourier domain
shifted_fft = image_fft .* shift_kernel;

% Step 4: Perform the inverse Fourier Transform to get the shifted image
shifted_image = real(ifft2(shifted_fft));

% Display the shifted image
figure;
imshow(shifted_image, []);
title('Shifted Image');

kernel_abs= real(shift_kernel);
figure
imshow(kernel_abs,[])

%% Q4-2

% Step 1: Read the image and extract the first slice
image = imread('S1_Q4_utils/ct.jpg');
image_gray = rgb2gray(image); % Convert to grayscale if necessary

% Display the original image and rotated image side by side
angle = 30; % Rotation angle in degrees
rotated_image = imrotate(image_gray, angle, 'bilinear', 'crop');

figure;
subplot(1, 2, 1);
imshow(image_gray, []);
title('Original Image');

subplot(1, 2, 2);
imshow(rotated_image, []);
title(['Rotated Image (', num2str(angle), ' Degrees)']);

% Step 2: Compute the Fourier Transforms of the original and rotated images
image_fft = fftshift(fft2(double(image_gray)));
rotated_image_fft = fftshift(fft2(double(rotated_image)));

% Compute the magnitude spectrum for visualization
original_fft_magnitude = log(1 + abs(image_fft));
rotated_fft_magnitude = log(1 + abs(rotated_image_fft));

% Display the Fourier Transforms side by side
figure;
subplot(1, 2, 1);
imshow(original_fft_magnitude, []);
title('Fourier Transform (Original Image)');

subplot(1, 2, 2);
imshow(rotated_fft_magnitude, []);
title(['Fourier Transform (Rotated Image - ', num2str(angle), ' Degrees)']);

% Step 3: Display the rotated Fourier domain
% The rotated Fourier Transform is already computed in 'rotated_image_fft'.
% We will directly plot its magnitude spectrum again for clarity.

figure;
imshow(log(1 + abs(rotated_image_fft)), []);
title(['Rotated in Fourier Domain (', num2str(angle), ' Degrees)']);

% Step 4: Perform the inverse Fourier Transform
inverse_fft_image = ifft2(ifftshift(rotated_image_fft));

% Display the inverse Fourier Transform result
figure;
imshow(real(inverse_fft_image), []);
title('Inverse Fourier Transform (Reconstructed Image from Rotated Fourier Domain)');

%% Q5

% Step 1: Read the Image
image = imread('S1_Q5_utils/t1.jpg');
image_gray = rgb2gray(image); % Convert to grayscale if necessary

% Display the original image
figure;
subplot(1, 4, 1);
imshow(image_gray, []);
title('Original Image');

% Step 2: Compute Central Differences Using circshift
% Vertical derivative (dy)
dy = circshift(image_gray, [-1, 0]) - circshift(image_gray, [1, 0]);

% Horizontal derivative (dx)
dx = circshift(image_gray, [0, -1]) - circshift(image_gray, [0, 1]);

% Display the vertical derivative
subplot(1, 4, 2);
imshow(dy, []);
title('Vertical Derivative (dy)');

% Display the horizontal derivative
subplot(1, 4, 3);
imshow(dx, []);
title('Horizontal Derivative (dx)');

% Step 3: Compute Gradient Magnitude
gradient_magnitude = sqrt(double(dx).^2 + double(dy).^2);

% Display the gradient magnitude
subplot(1, 4, 4);
imshow(gradient_magnitude, []);
title('Gradient Magnitude');

%% Q6x

% Step 1: Read the image
image = imread('S1_Q5_utils/t1.jpg');
image_gray = rgb2gray(image); % Convert to grayscale if necessary

% Display the original image
figure;
imshow(image_gray, []);
title('Original Image');

% Step 2: Apply Sobel edge detection
sobel_edges = edge(image_gray, 'sobel');

% Display Sobel edge detection result
figure;
imshow(sobel_edges, []);
title('Sobel Edge Detection');

% Step 3: Apply Canny edge detection
canny_edges = edge(image_gray, 'canny');

% Display Canny edge detection result
figure;
imshow(canny_edges, []);
title('Canny Edge Detection');

% Step 4: Compare with previous gradient magnitude
% Compute gradient magnitude using central difference (reuse previous code)
dx = circshift(image_gray, [0, -1]) - circshift(image_gray, [0, 1]);
dy = circshift(image_gray, [-1, 0]) - circshift(image_gray, [1, 0]);
gradient_magnitude = sqrt(double(dx).^2 + double(dy).^2);

% Display the comparison of all methods
figure;
subplot(1, 3, 1);
imshow(gradient_magnitude, []);
title('Gradient Magnitude');

subplot(1, 3, 2);
imshow(sobel_edges, []);
title('Sobel Method');

subplot(1, 3, 3);
imshow(canny_edges, []);
title('Canny Method');


