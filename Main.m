clc
clear
close all

%%
[filename,pathname]=uigetfile('*.*','Select image');
inputImage=imread([pathname,filename]);
inputImage = imresize(inputImage,[400 400]);
figure
imshow(inputImage)
title('Input Image')

gray = rgb2gray(inputImage);
figure
imshow(gray)
title('Gray image')
                                                                    
Denoise = medfilt2(gray);
figure
imshow(Denoise)
title('Denoised Image')

adj = imadjust(Denoise);
figure
imshow(adj)
title('Adjusted Image')

% Choose a threshold value T (you can adjust this value)
Threshold = 150; % Adjust this threshold value as needed

% Apply thresholding to create the binary image
binaryImage = Denoise > Threshold;
figure
imshow(binaryImage);
title('Binary Image');

% Find contours in the binary image
contourImage = bwperim(binaryImage,4);
% Overlay the contours on the original image
segmentedImage = gray;
segmentedImage(contourImage) = 1; % Highlight contours in white
% Display the segmented image
figure
imshow(segmentedImage);
title('Segmented Image');

stats = regionprops(binaryImage,'Area');
allValues = [stats.Area];
maxValue = max(allValues);
bigBlobs = bwareafilt(binaryImage, [1, maxValue-1]);
figure('Name','ROI');imshow(bigBlobs)

GLCM = graycomatrix(bigBlobs); 
stats = graycoprops(GLCM,{'Contrast','Correlation','Energy','Homogeneity'});
all_features = [stats.Contrast, stats.Correlation, stats.Energy, stats.Homogeneity];

Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;

% Display the results
fprintf('Contrast: %f\n', Contrast);
fprintf('Correlation: %f\n', Correlation);
fprintf('Energy: %f\n', Energy);
fprintf('Homogeneity: %f\n', Homogeneity);

%% Classification of svm 

load Features
load labels

SVMModel = fitcecoc(Features,labels);
[YPred, s] = predict(SVMModel,all_features);
YPred = cell2mat(YPred); 
msgbox(YPred);

% Split the data into features (X) and labels (Y)
X = Features;
Y = labels;

% Split the data into training and testing sets
% Set random seed for reproducibility
cv = cvpartition(size(X, 1), 'HoldOut', 0.3); % 70% training, 30% testing
XTrain = X(cv.training,:);
YTrain = Y(cv.training,:);
XTest = X(cv.test,:);
YTest = Y(cv.test,:);

SVMModel = fitcknn(XTrain, YTrain);
YPred = predict(SVMModel, XTest);

% Evaluate the classifier's performance
accuracy = sum(strcmp(YPred, YTest)) / numel(YTest);
fprintf('The Classified Accuracy: %.2f%%\n', accuracy * 100);

% Create a confusion matrix
C = confusionmat(YTest, YPred);
figure
confusionchart(C, unique(YTest));
title('confusionchart')

