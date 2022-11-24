%% 1. Load data
styleImage = im2double(imread("sailing_boat_art.jpg"));
style_mask = load_mask("misty-mood-leonid-afremov-mask.jpg");
contentImage = imread("coloredDrawing3.jpg");
content_mask = load_mask("sagano_bamboo_forest_mask.jpg");

figure(1)
imshow(imtile({styleImage,contentImage},BackgroundColor="w"));
saveas(figure(1), 'coloredDrawing3_sailing_boat.jpg')


%% 2. Load and modify VGG-19 network

net = vgg19;
lastFeatureLayerIdx = 38;
layers = net.Layers;
layers = layers(1:lastFeatureLayerIdx);

for l = 1:lastFeatureLayerIdx
    layer = layers(l);
    if isa(layer,"nnet.cnn.layer.MaxPooling2DLayer")
        layers(l) = averagePooling2dLayer(layer.PoolSize,Stride=layer.Stride,Name=layer.Name);
    end
end

lgraph = layerGraph(layers);
%plot(lgraph)
%title("Feature Extraction Network")
dlnet = dlnetwork(lgraph);


%% 3. Preprocess Data
% imageSize = size(content);
% imageSize = [imageSize(1)/2, imageSize(2)/2];
imageSize = [256,192];
styleImg = imresize(styleImage,imageSize);
contentImg = imresize(contentImage,imageSize);
content_mask = imresize(content_mask, imageSize);
style_mask = imresize(style_mask, imageSize);

% Subtract the mean to get "zero center" normalization
imgInputLayer = lgraph.Layers(1);
meanVggNet = imgInputLayer.Mean(1,1,:);
styleImg = rescale(single(styleImg),0,255) - meanVggNet;
contentImg = rescale(single(contentImg),0,255) - meanVggNet;
style_mask = rescale(single(style_mask),0,255) - meanVggNet;
content_mask = rescale(single(content_mask),0,255) - meanVggNet;
% figure(3)
% imshow(styleImg)
% figure(4)
% imshow(contentImg)

%% 4. Initialize Transfer Image
noiseRatio = 0.7;
randImage = randi([-20, 20], [imageSize 3]);
transferImage = noiseRatio.*randImage + (1-noiseRatio).*contentImg;
%transferImage(find(maskImage == 0)) = contentImg(find(maskImage == 0));


%% 5. Define Loss Functions
% Creating a struct called "styleTransferOptions" with fields,
% contentFeatureLayerNames, contentFeatureLayerWeights,
% styleFeatureLayerNames, styleFeatureLayersWeights, alpha and beta.
% Content Loss
styleTransferOptions.contentFeatureLayerNames = ["conv4_2"]; %["conv4_2", "conv5_2"];
styleTransferOptions.contentFeatureLayerWeights = 1; %[1.0, 0.5];

% Style Loss
styleTransferOptions.styleFeatureLayerNames = ["relu3_1", "relu4_1"]; %["conv1_1","conv2_1","conv3_1","conv4_1","conv5_1"];
styleTransferOptions.styleFeatureLayerWeights = [1.5, 3.5]; %[0.5,1.0,1.5,3.0,4.0];

styleTransferOptions.alpha = 1; 
styleTransferOptions.beta = 1e3;

TEMP = 0;
TEMPT = 0;
%% 6. Train the network
% Use gradient descent and iterations
numIterations = 1000;
learningRate = 2;
trailingAvg = [];
trailingAvgSq = [];

% Convert all 3 images to dlarray
dlStyle = dlarray(styleImg,"SSC");
dlContent = dlarray(contentImg,"SSC");
dlTransfer = dlarray(transferImage,"SSC");

if canUseGPU
    dlContent = gpuArray(dlContent);
    dlStyle = gpuArray(dlStyle);
    dlTransfer = gpuArray(dlTransfer);
end

% Extract the content features from the content image
numContentFeatureLayers = numel(styleTransferOptions.contentFeatureLayerNames);
contentFeatures = cell(1,numContentFeatureLayers);
[contentFeatures{:}] = forward(dlnet,dlContent,Outputs=styleTransferOptions.contentFeatureLayerNames);

% Extract the style features from the style image
numStyleFeatureLayers = numel(styleTransferOptions.styleFeatureLayerNames);
styleFeatures = cell(1,numStyleFeatureLayers);
[styleFeatures{:}] = forward(dlnet,dlStyle,Outputs=styleTransferOptions.styleFeatureLayerNames);

% The actual training & iterations
figure
minimumLoss = inf;
i = 0;
for iteration = 1:numIterations
    i = i+1;
    % Evaluate the transfer image gradients and state using dlfeval and the
    % imageGradients function listed at the end of the example
    [grad,losses] = dlfeval(@imageGradients,dlnet,dlTransfer,contentFeatures,styleFeatures,styleTransferOptions);
    [dlTransfer,trailingAvg,trailingAvgSq] = adamupdate(dlTransfer,grad,trailingAvg,trailingAvgSq,iteration,learningRate);
  
    if losses.totalLoss < minimumLoss
        minimumLoss = losses.totalLoss;
        dlOutput = dlTransfer;        
    end   
    
    % Display the transfer image on the first iteration and after every 50
    % iterations. The postprocessing steps are described in the "Postprocess
    % Transfer Image for Display" section of this example
    if mod(iteration,50) == 0 || (iteration == 1)
        
        transferImage = gather(extractdata(dlTransfer));
        transferImage = transferImage + meanVggNet;
        transferImage = uint8(transferImage);
        transferImage = imresize(transferImage,size(contentImage,[1 2]));
        figure(ceil(i/50)+1)
        image(transferImage)
        title(["Transfer Image After Iteration ",num2str(iteration)])
        axis off image
        drawnow
        saveas(figure(ceil(i/50)+1), ['coloredDrawing3_sailing_boat_iteration',  num2str(i), '.jpg'])
    end   
    
end


function [gradients,losses] = imageGradients(dlnet,dlTransfer,contentFeatures,styleFeatures,params)
 
    % Initialize transfer image feature containers
    numContentFeatureLayers = numel(params.contentFeatureLayerNames);
    numStyleFeatureLayers = numel(params.styleFeatureLayerNames);
 
    transferContentFeatures = cell(1,numContentFeatureLayers);
    transferStyleFeatures = cell(1,numStyleFeatureLayers);
 
    % Extract content features of transfer image
    [transferContentFeatures{:}] = forward(dlnet,dlTransfer,Outputs=params.contentFeatureLayerNames);
     
    % Extract style features of transfer image
    [transferStyleFeatures{:}] = forward(dlnet,dlTransfer,Outputs=params.styleFeatureLayerNames);
 
    % Calculate content loss
    cLoss = contentLoss(transferContentFeatures,contentFeatures,params.contentFeatureLayerWeights);
 
    % Calculate style loss
    sLoss = newstyleLoss(transferStyleFeatures,styleFeatures,params.styleFeatureLayerWeights);
 
    % Calculate final loss as weighted combination of content and style loss 
    loss = (params.alpha * cLoss) + (params.beta * sLoss);
 
    % Calculate gradient with respect to transfer image
    gradients = dlgradient(loss,dlTransfer);
    
    % Extract various losses
    losses.totalLoss = gather(extractdata(loss));
    losses.contentLoss = gather(extractdata(cLoss));
    losses.styleLoss = gather(extractdata(sLoss));
 
end

function loss = contentLoss(transferContentFeatures,contentFeatures,contentWeights)

    loss = 0;
    for i=1:numel(contentFeatures)
        temp = 0.5 .* mean((transferContentFeatures{1,i} - contentFeatures{1,i}).^2,"all");
        loss = loss + (contentWeights(i)*temp);
    end
end

function loss = newstyleLoss(transferStyleFeatures,styleFeatures,styleWeights)
%NEWSTYLELOSS creates the loss value between the style image and generated
%image and the mask image as well
    loss = 0;
    for i=1:numel(styleFeatures)
        
        tsf = transferStyleFeatures{1,i};
        sf = styleFeatures{1,i};    
        [h,w,c] = size(sf);
        log_ = log2(min(h,w));

        if log_ > log2(64)
            res = 16;
        else
            res = gcd(h,w);
        end
%         TEMP = sf(:,:,1);
%         TEMPT = tsf(:,:,1);
        patch_size = [h/res, w/res];
        euclidean_norm = zeros(c);
        for j = 1:c
            sf_part = sf(:,:,j);
            tsf_part1 = tsf(:,:,j);
            tsf_part = extractdata(tsf_part1);
            euclidean_norm(j) = calculateEuclideanNorm(sf_part,tsf_part,patch_size, res);

        end

        %gramStyle = calculateGramMatrix(sf);
        %gramTransfer = calculateGramMatrix(tsf);
        %sLoss = mean((gramTransfer - gramStyle).^2,"all") / ((h*w*c)^2);
        

        %patches_difference = tsf_patches - sf_patches2;
%         cross_correlation = sum(patches)
        
        sLoss = mean(euclidean_norm,"all");
        loss = loss + (styleWeights(i)*sLoss);
    end
    loss = dlarray(loss)
end



function loss = styleLoss(transferStyleFeatures,styleFeatures,styleWeights)
% I think transferStyleFeatures is styleFeatures (the 5 conv layers) of the
% generated transfer image, while styleFeatures is of the style image itself.
    loss = 0;
    for i=1:numel(styleFeatures)
        
        tsf = transferStyleFeatures{1,i};
        sf = styleFeatures{1,i};    
        [h,w,c] = size(sf);
        
        gramStyle = calculateGramMatrix(sf);
        gramTransfer = calculateGramMatrix(tsf);
        sLoss = mean((gramTransfer - gramStyle).^2,"all") / ((h*w*c)^2);
        
        loss = loss + (styleWeights(i)*sLoss);
    end
end

function gramMatrix = calculateGramMatrix(featureMap)
    [H,W,C] = size(featureMap);
    reshapedFeatures = reshape(featureMap,H*W,C);
    gramMatrix = reshapedFeatures' * reshapedFeatures;
end

function euclidean_norm = calculateEuclideanNorm(sf, tsf, patch_size, res)
    % Local patching (based on Li and Wand):
    %sf_size = size(sf)
    sf_patches = im2col(sf, patch_size, 'distinct');    %768 x 256
    tsf_patches = im2col(tsf, patch_size, 'distinct'); %768 x 256
    rng(1);
    tsf_patches = tsf_patches + rand(size(tsf_patches));
    % Do normalized cross-correlation between each local patches
    % transfer image's feature map and the style feature map
    num_of_patches = size(tsf_patches);
    num_of_patches = num_of_patches(2); %256
    %strongest_patch = zeros(num_of_patches);
    sf_matches = zeros(size(tsf_patches));
    patches_difference = zeros(size(tsf_patches));
    euclidean_norm_matrix = zeros(1,num_of_patches(1));
    for i = 1:num_of_patches
        patch = tsf_patches(:,i);
        patch = round(reshape(patch, patch_size));
        %sf_map = reshape(sf, sf_size);
        %patch = imresize(patch, [h,w,c]);
        sf_ = extractdata(sf);
        cross_correlation = normxcorr2(patch, sf_);
        %cc_size = size(cross_correlation)
        cross_correlation = cross_correlation(patch_size(1)+1:end, patch_size(2)+1:end);
        %cc_size = size(cross_correlation)
        [ypeak,xpeak] = find(cross_correlation==max(cross_correlation(:)));
        %yoffSet = ypeak-size(patch,1)
        %xoffSet = xpeak-size(patch,2)
        y = ceil(ypeak(1) / patch_size(2));
        x = ceil(xpeak(1) / patch_size(1));
        strongest_patch = round(x*res + y);
        sf_matches(:,i) = sf_patches(:, strongest_patch);
        patches_difference(i) = tsf_patches(i) - sf_matches(i);
        euclidean_norm_matrix(i) = norm(patches_difference(i));
    end
    euclidean_norm = sum(euclidean_norm_matrix.^2)
end