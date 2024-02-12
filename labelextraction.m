clear all;close all;clc;
srcFiles = dir('C:\Users\0659\Desktop\TK122459\Dataset\Normal cases\*.jpg');  % the folder in which ur images exists
for i = 1 : length(srcFiles)
filename = strcat('C:\Users\0659\Desktop\TK122459\Dataset\Normal cases\',srcFiles(i).name);

load labels.mat
labels(30+i,:) = {'Normal'};
save('labels','labels')
srcFiles(i).name
end
