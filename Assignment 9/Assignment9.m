clc; clear;

imagePath = 'S3_Q1_utils/thorax_t1.jpg'; % Replace with the correct path
clickToGetCoordinates(imagePath);

%% Q1

originalImage = imread(imagePath);
image = double(rgb2gray(originalImage));

figure;
imshow(originalImage);
title('Original Image');

figure;
left_lung = mask_image(image, 90, 90, 35, 40);
right_lung = mask_image(image, 180, 90, 35, 40);
lung = left_lung | right_lung;
Overlay(image, lung)

figure;
liver1 = mask_image(image, 122, 148, 25, 30);
liver2 = mask_image(image, 90, 142, 25, 30);
liver3 = mask_image(image, 81, 173, 25, 30);
liver = liver1 | liver2 | liver3;
Overlay(image, liver)

%% Q2

% Load and preprocess the images
img1 = imread('S3_Q2_utils/t1.jpg'); % Replace with actual image file path
img2 = imread('S3_Q2_utils/t2.jpg');
img3 = imread('S3_Q2_utils/pd.jpg');

% Convert images to grayscale and double for processing
img1 = double(rgb2gray(img1));
img2 = double(rgb2gray(img2));
img3 = double(rgb2gray(img3));

% Concatenate the pixel intensities into a single feature matrix
[rows, cols] = size(img1);
featureMatrix = [img1(:), img2(:), img3(:)]; % Each row is a pixel, columns are intensities from 3 images

% Apply k-means clustering
numClusters = 6; % Number of clusters
[clusterLabels, ~] = kmeans(featureMatrix, numClusters, 'MaxIter', 1000);

% Reshape the cluster labels back into image dimensions
clusteredImg = reshape(clusterLabels, rows, cols);

% Visualize the segmented clusters
figure;
for k = 1:numClusters
    subplot(2, 3, k);
    imshow(clusteredImg == k, []);
    title(['Cluster ', num2str(k)]);
end

%% Q3

maxIterations = 100;

% Apply manual K-means
manualClusterLabels = manualKMeans(featureMatrix, numClusters, maxIterations);

% Reshape manual cluster labels into image dimensions
manualClusteredImg = reshape(manualClusterLabels, rows, cols);

% Apply MATLAB's built-in kmeans
builtinClusterLabels = kmeans(featureMatrix, numClusters, 'MaxIter', maxIterations);

% Reshape MATLAB's cluster labels into image dimensions
builtinClusteredImg = reshape(builtinClusterLabels, rows, cols);

% Compare the results visually
figure;

% Manual K-means clusters
for k = 1:numClusters
    subplot(2, numClusters, k);
    imshow(manualClusteredImg == k, []);
    title(['Manual Cluster ', num2str(k)]);
end

% Built-in K-means clusters
for k = 1:numClusters
    subplot(2, numClusters, k + numClusters);
    imshow(builtinClusteredImg == k, []);
    title(['Built-in Cluster ', num2str(k)]);
end

%% Q4

[centers, U] = fcm(featureMatrix, numClusters); % FCM clustering

% Compute final cluster labels based on maximum membership values
[~, clusterLabels] = max(U); % U is the membership matrix (numClusters x numPoints)

% Reshape cluster labels into image dimensions
fcmClusteredImg = reshape(clusterLabels, rows, cols);

% Visualize the FCM segmented clusters
figure;
for k = 1:numClusters
    subplot(2, 3, k);
    imshow(fcmClusteredImg == k, []);
    title(['FCM Cluster ', num2str(k)]);
end


%%

function Overlay(f, mask)

    m = max(f(:));
    fr = f;
    fg = f + mask / max(mask(:)) * m/2;
    fb = f;
   
    imshow(reshape([fr fg fb],[size(f,1) size(f,2) 3])/m, []);
    
    drawnow;

end

function clickToGetCoordinates(imagePath)
    % Function to display an image and allow the user to click on it to get coordinates.
    % Input: 
    %   imagePath - Path to the image file.
    
    % Read the image
    img = imread(imagePath);
    
    % Display the image
    figure;
    imshow(img);
    title('Click on the image to get coordinates (Close the figure to end)');
    
    % Instructions
    disp('Click on the image to get coordinates. Close the figure window to stop.');
    
    % Loop to capture multiple clicks
    while true
        try
            % Get the coordinates of a click
            [x, y] = ginput(1); % Get one point at a time
            
            % Display the coordinates
            fprintf('You clicked at: (X: %.2f, Y: %.2f)\n', x, y);
        catch
            % Break the loop if the figure is closed
            disp('Figure closed. Exiting coordinate selection.');
            break;
        end
    end
end

function mask = mask_image(image, y0, x0, loc_threshold, color_tresh)
    [rows, cols] = size(image);
    mask = zeros(size(image));
        
    xmin = max(1, x0 - loc_threshold); 
    xmax = min(rows, x0 + loc_threshold); 
    ymin = max(1, y0 - loc_threshold); 
    ymax = min(cols, y0 + loc_threshold); 
    
    thresh = image(x0, y0);
    
    for i = xmin:xmax
        for j = ymin:ymax
            if (image(i, j) <= thresh + color_tresh && image(i, j) >= thresh - color_tresh)
                mask(i, j) = 1; 
            end
        end
    end
end

function clusterLabels = manualKMeans(data, numClusters, maxIterations)
    % Manual implementation of k-means clustering
    % Inputs:
    %   data - N x D matrix where each row is a data point (pixels in this case)
    %   numClusters - Number of clusters to create
    %   maxIterations - Maximum number of iterations
    % Outputs:
    %   clusterLabels - N x 1 vector of cluster labels for each data point

    % Number of data points
    numPoints = size(data, 1);

    % Randomly initialize cluster centers
    clusterCenters = data(randi(numPoints, numClusters, 1), :);

    % Initialize cluster labels
    clusterLabels = zeros(numPoints, 1);

    for iter = 1:maxIterations
        % Step 1: Assign each data point to the nearest cluster center
        for i = 1:numPoints
            distances = sum((data(i, :) - clusterCenters).^2, 2); % Squared Euclidean distance
            [~, clusterLabels(i)] = min(distances); % Assign to the nearest cluster
        end

        % Step 2: Update the cluster centers
        newClusterCenters = zeros(numClusters, size(data, 2));
        for k = 1:numClusters
            pointsInCluster = data(clusterLabels == k, :); % Points belonging to cluster k
            if ~isempty(pointsInCluster)
                newClusterCenters(k, :) = mean(pointsInCluster, 1); % Compute new center
            else
                % If a cluster has no points, reinitialize it randomly
                newClusterCenters(k, :) = data(randi(numPoints), :);
            end
        end

        % Check for convergence (if cluster centers do not change)
        if all(abs(newClusterCenters - clusterCenters) < 1e-5, 'all')
            break;
        end

        % Update cluster centers for next iteration
        clusterCenters = newClusterCenters;
    end
end


