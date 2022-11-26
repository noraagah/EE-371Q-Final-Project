%% 1. Load data
styleImage = im2double(imread("Monet.jpg"));
style_mask = imread("Monet_sem.png");
contentImage = imread("Coastline_sem.png");
content_mask = imread("Coastline_sem.png");

figure(1)
imshow(imtile({styleImage,contentImage},BackgroundColor="w"));
saveas(figure(1), 'Coastline_Monet.jpg')

% numClusters = 2;
% [L,Centers] = imsegkmeans(contentImage,numClusters); % not sure how many clusters yet
% B = labeloverlay(contentImage,L);
% B = imbinarize(B(:,:,3));
% figure(2)
% imshow(B);
% title("Labeled Image")
% 
% styleImg = imread("misty-mood-leonid-afremov.jpg");
% [L,Centers] = imsegkmeans(styleImg,numClusters); % not sure how many clusters yet
% B = labeloverlay(styleImg,L);
% B = imbinarize(B(:,:,1));
% figure(3)
% imshow(B)
% title("Labeled Image")

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
%imageSize = size(contentImage);
imageSize = [384, 512];
imageSize = imageSize/2;
styleImg = imresize(styleImage,imageSize);
contentImg = imresize(contentImage,imageSize);
content_mask = imresize(content_mask, imageSize);
style_mask = imresize(style_mask, imageSize);

numOfRegions = 6;
[contentMasks, styleMasks] = extractMask(im2gray(content_mask), im2gray(style_mask), numOfRegions);

%%
% Subtract the mean to get "zero center" normalization
imgInputLayer = lgraph.Layers(1);
meanVggNet = imgInputLayer.Mean(1,1,:);
styleImg = rescale(single(styleImg),0,255) - meanVggNet;
contentImg = rescale(single(contentImg),0,255) - meanVggNet;
% style_mask = rescale(single(style_mask),0,255) - meanVggNet;
% content_mask = rescale(single(content_mask),0,255) - meanVggNet;
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
styleTransferOptions.styleFeatureLayerNames = ["conv1_1","conv2_1","conv3_1","conv4_1","conv5_1"];
styleTransferOptions.styleFeatureLayerWeights = [0.5,1.0,1.5,3.0,4.0];

styleTransferOptions.maskFeatureLayerNames = ["conv1_1","conv2_1","conv3_1","conv4_1","conv5_1"];
styleTransferOptions.maskFeatureLayerWeights = [0.5, 0.5, 0.5, 0.5, 0.5];

styleTransferOptions.alpha = 1; 
styleTransferOptions.beta = 100;


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

% Convert masks to dlarray
dlStyleMask = cell(1,numOfRegions); % cells of dlarrays for each region
dlContentMask = cell(1, numOfRegions); % cells of dlarrays for each region
for i = 1:numOfRegions
    temp = contentMasks(:,:,i);
    dlContentMask{i} = dlarray(temp, "SSC");
    temp = styleMasks(:,:,i);
    %imshow(temp);
    dlStyleMask{i} = dlarray(temp, "SSC");
end
% dlStyleMask = dlarray(styleMasks, "SSC");
% dlContentMask = dlarray(contentMasks, "SSC");

% if canUseGPU
%     dlContent = gpuArray(dlContent);
%     dlStyle = gpuArray(dlStyle);
%     dlTransfer = gpuArray(dlTransfer);
% end

% Extract the content features from the content image
numContentFeatureLayers = numel(styleTransferOptions.contentFeatureLayerNames);
contentFeatures = cell(1,numContentFeatureLayers);
[contentFeatures{:}] = forward(dlnet,dlContent,Outputs=styleTransferOptions.contentFeatureLayerNames);

% Extract the style features from the style image
numStyleFeatureLayers = numel(styleTransferOptions.styleFeatureLayerNames);
styleFeatures = cell(1,numStyleFeatureLayers);
[styleFeatures{:}] = forward(dlnet,dlStyle,Outputs=styleTransferOptions.styleFeatureLayerNames);

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


%%
% The actual training & iterations
figure
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
        saveas(figure(ceil(i/50)+1), ['Coastline_Monet_Neuraldoodle_iteration',  num2str(i), '.jpg'])
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
    regionalWeightFactor = 0.1;
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
% Won't pass the mask guides through CNNs, but rather will just resize(/2
% for each layer).
    mask_size = size(contentMask,2);
    % To hold the F^r_l for each r=1:mask_size and l=1:numel(styleFeatures)
    %content_guidedGramMatrix = cell(mask_size,numel(styleFeatures));
    %style_guidedGramMatrix = cell(mask_size,numel(styleFeatures));
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
            %content_guidedGramMatrix{r,i} = content_guide(:) .* reshaped_sf;
            %style_guidedGramMatrix{r,i} = style_guide(:) .* reshaped_tsf;
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

% function extracted_masks = extractMask(contentMask, styleMask, r)
%     % Turn it into grayscale
% %     mask_grayscale = im2gray(mask);
% %     mask_filtered = func_median(mask_grayscale);
%     content_graylevels = zeros(1,r);
%     style_graylevels = zeros(1,r);
%     content_temp = contentMask;
%     style_temp = styleMask;
%     sizes = size(contentMask);
% %     content_masks = zeros(sizes(1), sizes(2), r);
% %     style_masks = zeros(sizes(1), sizes(2), r);
%     extracted_masks = zeros(sizes(1), sizes(2), r);
%     for i = 1:r
%         maj = mode(content_temp(:));
%         content_graylevels(i) = maj;
%         content_temp(content_temp == maj) = NaN;
% 
%         maj = mode(style_temp(:));
%         style_graylevels(i) = maj;
%         style_temp(style_temp == maj) = NaN;
%     end
%     for i = 1:r
%         m = uint8(content_graylevels(i));
%         temp1 = content_temp;
%         temp1(temp1 ~= m) = 0;
%         temp1(temp1 == m) = 1;
%         
%         m = uint8(style_graylevels(i));
%         temp2 = style_temp;
%         temp2(temp2 ~= m) = 0;
%         temp2(temp2 == m) = 1;
% 
%         temp = temp1 .* temp2;
%         extracted_masks(:,:,i) = temp;
%     end
% end
