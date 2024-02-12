clear all;close all;clc;
srcFiles = dir('C:\Users\0659\Desktop\TK122459\Dataset\New folder\*.jpg');  % the folder in which ur images exists
Features = zeros(length(srcFiles), 4);
for i = 1 : length(srcFiles)
filename = strcat('C:\Users\0659\Desktop\TK122459\Dataset\New folder\',srcFiles(i).name);
inputImage = imread(filename);
inputImage = imresize(inputImage,[400 400]);
% figure
% imshow(inputImage)
% title('Input Image')

gray = rgb2gray(inputImage);
% figure
% imshow(gray)
% title('Gray image')
                                   
Denoise = medfilt2(gray);
% figure
% imshow(Denoise)
% title('Denoised Image')

adj = imadjust(Denoise);
% figure
% imshow(adj)
% title('Adjusted Image')

% Choose a threshold value T (you can adjust this value)
Threshold = 150; % Adjust this threshold value as needed

% Apply thresholding to create the binary image
binaryImage = Denoise > Threshold;
% figure
% imshow(binaryImage);
% title('Binary Image');

% Find contours in the binary image
contourImage = bwperim(binaryImage,4);
% Overlay the contours on the original image
segmentedImage = gray;
segmentedImage(contourImage) = 1; % Highlight contours in white
% Display the segmented image
% figure
% imshow(segmentedImage);
% title('Segmented Image');

stats = regionprops(binaryImage,'Area');
allValues = [stats.Area];
maxValue = max(allValues);
bigBlobs = bwareafilt(binaryImage, [1, maxValue-1]);
% figure('Name','ROI');imshow(bigBlobs)

GLCM = graycomatrix(bigBlobs); 
stats = graycoprops(GLCM,{'Contrast','Correlation','Energy','Homogeneity'});
features = [stats.Contrast, stats.Correlation, stats.Energy, stats.Homogeneity];


%%
load Features.mat
Features(15+i,:) = features;
save('Features','Features')
srcFiles(i).name
close all

end
