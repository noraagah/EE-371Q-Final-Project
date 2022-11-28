%% Neural Style Transfer with Semantic Masks
% This code shows how to apply the stylistic appearance of one image to the 
% scene content of a colored doodle image using a pretrained VGG-19 network.

% Neural Style Transfer Using Deep Learning
% This example shows how to apply the stylistic appearance of one image to the 
% scene content of a second image using a pretrained VGG-19 network.
% 
% Code from: https://www.mathworks.com/help/images/neural-style-transfer-using-deep-learning.html

%% Adapted the code with the following additions:
% 1. Added content and style maps
% 2. Wrote the support function, extractMask(), to extract regional mask
% image maps in order to do semantic map style transfer
% 3. Wrote a new style loss function that incorporated the regional mask
% image maps when calculating the gram matrix for similarities
%% 1. Load data
styleImage_filename = "beach_near_etretat.jpg";
styleMask_filename = "beach_near_etretat_sem.png";
contentImage_filename = "doodle1.png";
contentMask_filename = "doodle1.png";
styleImage = im2double(imread(styleImage_filename));
style_mask = imread(styleMask_filename);
contentImage = imread(contentImage_filename);
content_mask = imread(contentMask_filename);

% new variable
numOfRegions = 6; % the number of colors used to draw the doodle & create the style mask

content_style_filename = 'doodle1_and_beach';
figure(1);
imshow(imtile({styleImage,contentImage},BackgroundColor="w"));
saveas(figure(1), [content_style_filename, '.jpg']);

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
dlnet = dlnetwork(lgraph);
% dlnet.Layers

%% 3. Preprocess Data

% Resize the images to a suitable size for running through the CNN
imageSize = [384, 512];
%imageSize = imageSize/2;
styleImg = imresize(styleImage,imageSize);
contentImg = imresize(contentImage,imageSize);
content_mask = imresize(content_mask, imageSize);
style_mask = imresize(style_mask, imageSize);

% Extract the region masks 1, ..., R for content and style mask, using the
% newly written function extractMask() 
[contentMasks, styleMasks] = extractMask(content_mask, style_mask, numOfRegions); 

% Subtract the mean to get "zero center" normalization
imgInputLayer = lgraph.Layers(1);
meanVggNet = imgInputLayer.Mean(1,1,:);
styleImg = rescale(single(styleImg),0,255) - meanVggNet;
contentImg = rescale(single(contentImg),0,255) - meanVggNet;

%% 4. Initialize Transfer Image 

noiseRatio = 0.7;
randImage = randi([-20, 20], [imageSize 3]);
transferImage = noiseRatio.*randImage + (1-noiseRatio).*contentImg; % initalize the transfer image using the original contentImage

%% 5. Define Loss Functions
% Creating a struct called "styleTransferOptions" with fields:
% contentFeatureLayerNames, contentFeatureLayerWeights, styleFeatureLayerNames, styleFeatureLayersWeights, alpha and beta.

% Content Loss
styleTransferOptions.contentFeatureLayerNames = ["conv4_2"]; %["conv4_2", "conv5_2"];
styleTransferOptions.contentFeatureLayerWeights = 1; %[1.0, 0.5];

% Style Loss
styleTransferOptions.styleFeatureLayerNames = ["conv1_1","conv2_1","conv3_1","conv4_1","conv5_1"];
styleTransferOptions.styleFeatureLayerWeights = [0.5,1.0,1.5,3.0,4.0];

% Attempt to pass the masks through the CNN
% styleTransferOptions.maskFeatureLayerNames = ["conv1_1","conv2_1","conv3_1","conv4_1","conv5_1"];
% styleTransferOptions.maskFeatureLayerWeights = [0.5, 0.5, 0.5, 0.5, 0.5];

styleTransferOptions.alpha = 1; 
styleTransferOptions.beta = 1000;


%% 6. Train the network
%% Declare variables for training
% Use gradient descent and iterations
numIterations = 1000;
learningRate = 2;
trailingAvg = [];
trailingAvgSq = [];

% Convert all 3 images to dlarray
dlStyle = dlarray(styleImg,"SSC");
dlContent = dlarray(contentImg,"SSC");
dlTransfer = dlarray(transferImage,"SSC");

% Convert masks to dlarray
dlStyleMask = cell(1,numOfRegions); % cells of dlarrays for each region mask of style image
dlContentMask = cell(1, numOfRegions); % cells of dlarrays for each region mask of content image
for i = 1:numOfRegions
    temp = contentMasks(:,:,i);
    dlContentMask{i} = dlarray(temp, "SSC");
    temp = styleMasks(:,:,i);
    %imshow(temp);
    dlStyleMask{i} = dlarray(temp, "SSC");
end

% dlStyleMask = dlarray(styleMasks, "SSC");
% dlContentMask = dlarray(contentMasks, "SSC");

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

% Attempt to pass the masks through the CNN
% Extract the mask guidance channels from the mask images

% numMaskFeatureLayers = numStyleFeatureLayers;
% styleMaskChannels = cell(numOfRegions,numMaskFeatureLayers);
% contentMaskChannels = cell(numOfRegions,numMaskFeatureLayers);
% for i = 1:numOfRegions
%     dlStyle_Mask = dlStyleMask{i};
%     dlContent_Mask = dlContentMask{i};
%     [styleMaskChannels{i,:}] = forward(dlnet,dlStyle_Mask,Outputs=styleTransferOptions.maskFeatureLayerNames);
%     [contentMaskChannels{i,:}] = forward(dlnet,dlContent_Mask,Outputs=styleTransferOptions.maskFeatureLayerNames);
% end
% [styleMaskFeatures{:}] = forward(dlnet,dlStyle_Mask,Outputs=styleTransferOptions.maskFeatureLayerNames);
% [contentMaskFeatures{:}] = forward(dlnet,dlContentMask,Outputs=styleTransferOptions.maskFeatureLayerNames);

%% The actual training iterations using gradient descent loss

minimumLoss = inf;
i = 0;
for iteration = 1:numIterations
    i = i+1;

    % Evaluate the transfer image gradients and state using dlfeval and the
    % imageGradients function listed at the end of the example

    % Calculate the gradients and the losses
    [grad,losses] = dlfeval(@imageGradients,dlnet,dlTransfer,contentFeatures,styleFeatures,dlContentMask, dlStyleMask, styleTransferOptions);
    
    % Update the weights and the transfer image
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
        saveas(figure(ceil(i/50)+1), ['content_style_filename', '_iteration', num2str(i), '.jpg'])
    end   
    
end


function [gradients,losses] = imageGradients(dlnet,dlTransfer,contentFeatures,styleFeatures,dlContentMask, dlStyleMask,params)
 
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
    sLoss1 = styleLoss(transferStyleFeatures,styleFeatures,params.styleFeatureLayerWeights);
    regionalWeightFactor = 0.3;
    sLoss2 = newStyleLoss(transferStyleFeatures, styleFeatures, dlContentMask, dlStyleMask, params.styleFeatureLayerWeights, regionalWeightFactor);
    sLoss = sLoss1+regionalWeightFactor*(sLoss2);

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

function loss = newStyleLoss(transferStyleFeatures, styleFeatures, contentMask, styleMask, styleWeights, regionalWeightFactor)
%NEWSTYLELOSS creates the loss value between the style image and generated
%image and the mask image as well
    loss = 0;
% Won't pass the mask guides through CNNs, but rather will just resize the
% masks to the size of the feature maps for each layer.
    mask_size = size(contentMask,2);
    % Calculate the new feature maps, F^r_l, for each r=1:mask_size and l=1:numel(styleFeatures),
    for r=1:mask_size
        % Get the guidance layer channels, T_l^r and normalize it
        content_guide = extractdata(contentMask{r}); % double
        %content_guide = content_guide / sqrt(diag(calculateGramMatrix(content_guide)));
        style_guide = extractdata(styleMask{r});
        %style_guide = style_guide / sqrt(diag(calculateGramMatrix(style_guide)));
        class(content_guide);
        for i = 1:numel(styleFeatures)
            size_l = size(content_guide) / (2^(i-1));
            size_l = size_l(1:2);
            % Resize the guides, "downsample"
            content_guide_resized = imresize(content_guide, size_l); %dlarray
            style_guide_resized = imresize(style_guide, size_l); %dlarray

            sf = styleFeatures{1,i};
            tsf = transferStyleFeatures{1,i};
            [H,W,C] = size(sf);
            reshaped_sf= reshape(sf,H*W,C);
            reshaped_tsf = reshape(tsf, H*W, C);

            % Multiply the feature maps of each layer with the R guidance channels
            style_guided_sf = style_guide_resized(:) .* reshaped_sf;
            content_guided_tsf = content_guide_resized(:) .* reshaped_tsf;

            style_guided_sf = reshape(style_guided_sf, [H,W,C]);
            content_guided_tsf = reshape(content_guided_tsf, [H,W,C]);

            % Compute the mask guided Gram Matrix for layer l and add it to
            % the overall loss
            gramStyle = calculateGramMatrix(style_guided_sf);
            gramTransfer = calculateGramMatrix(content_guided_tsf);
            sLoss = mean((gramTransfer - gramStyle).^2,"all") / ((H*W*C)^2);
        
            loss = loss + (styleWeights(i)*sLoss);
        end
    end
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

function [content_masks, style_masks] = extractMask(contentMask, styleMask, r)
% EXTRACTMASK returns the region guidance maps for the content and the
% style masks. The inputs are grayscale images of the content mask and the
% style mask.

    contentMask = im2gray(contentMask);
    styleMask = im2gray(styleMask);
    content_graylevels = zeros(1,r);
    style_graylevels = zeros(1,r);
    content_temp = double(contentMask);
    style_temp = double(styleMask);
    size1 = size(contentMask);
    content_masks = zeros(size1(1), size1(2), r);
    style_masks = zeros(size1(1), size1(2), r);
    extracted_masks = zeros(size1(1), size1(2), r);
    for i = 1:r
        maj = mode(content_temp(:));
        content_graylevels(i) = maj;
        content_temp(content_temp == maj) = NaN;
    % Assume the style_graylevels are the same as the content_graylevels!
    end

    for i = 1:r
        m = uint8(content_graylevels(i));
        temp1 = contentMask;
        temp1(temp1 ~= m) = 255;
        temp1(temp1 == m) = 1;
        temp1(temp1 == 255) = 0;
        content_masks(:,:,i) = temp1;

        m = uint8(content_graylevels(i));
        temp2 = styleMask;
        temp2(temp2 ~= m) = 255;
        temp2(temp2 == m) = 1;
        temp2(temp2 == 255) = 0;
        style_masks(:,:,i) = temp2;
    end
end
