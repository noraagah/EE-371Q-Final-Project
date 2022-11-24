%% 1. Load data
styleImage = im2double(imread("starryNight.jpg"));
style_mask = load_mask("starryNight_mask.jpg");
contentImage = imread("drawing1.jpg");
content_mask = load_mask("drawing1_mask.jpg");

figure(1)
imshow(imtile({styleImage,contentImage},BackgroundColor="w"));
saveas(figure(1), 'drawing1_starryNight.jpg')

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

styleMasks = extractMask(style_mask);
contentMasks = extractMask(content_mask);

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
styleTransferOptions.contentFeatureLayerNames = ["conv4_2"]%["conv4_2", "conv5_2"];
styleTransferOptions.contentFeatureLayerWeights = 1 %[1.0, 0.5];

% Style Loss
styleTransferOptions.styleFeatureLayerNames = ["conv1_1","conv2_1","conv3_1","conv4_1","conv5_1"];
styleTransferOptions.styleFeatureLayerWeights = [0.5,1.0,1.5,3.0,4.0];

styleTransferOptions.alpha = 1; 
styleTransferOptions.beta = 1e3;


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
dlContentMask = dlarray(content_mask, "SSC");
dlStyleMask = dlarray(style_mask, "SSC");

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

% Extract the guidance channel layers from the mask images
numMaskFeatureLayers = numStyleFeatureLayers;
styleMaskFeatures = styleFeatures;
[styleMaskFeatures{:}] = forward(dlnet,dlStyleMask,Outputs=styleTransferOptions.styleFeatureLayerNames);
contentMaskFeatures = styleFeatures;
[contentMaskFeatures{:}] = forward(dlnet,dlContentMask,Outputs=styleTransferOptions.styleFeatureLayerNames);
%%
% The actual training & iterations
figure
minimumLoss = inf;
i = 0;
for iteration = 1:numIterations
    i = i+1;
    % Evaluate the transfer image gradients and state using dlfeval and the
    % imageGradients function listed at the end of the example
    [grad,losses] = dlfeval(@imageGradients,dlnet,dlTransfer,contentFeatures,styleFeatures,contentMaskFeatures,styleTransferOptions);
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
        saveas(figure(ceil(i/50)+1), ['bamboo_forest_misty_mood_masked1_iteration',  num2str(i), '.jpg'])
    end   
    
end


function [gradients,losses] = imageGradients(dlnet,dlTransfer,contentFeatures,styleFeatures,contentMaskFeatures,params)
 
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
    %sLoss = styleLoss(transferStyleFeatures,styleFeatures,params.styleFeatureLayerWeights);
    regionalWeightFactor = 1;
    sLoss = newStyleLoss(transferStyleFeatures, styleFeatures, contentMaskFeatures, params.styleFeatureLayerWeights, regionalWeightFactor);

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

function loss = newStyleLoss(transferStyleFeatures, styleFeatures, contentMaskFeatures, styleWeights, regionalWeightFactor)
%NEWSTYLELOSS creates the loss value between the style image and generated
%image and the mask image as well
    loss = 0;

    for i=1:numel(contentMaskFeatures)
        cmf = contentMaskFeatures{1,i};
        sf = styleFeatures{1,i};
        tsf = transferStyleFeatures{1,i};
        sfr = cmf .* sf;
        tsfr = cmf .* tsf;
        [h,w,c] = size(sf);
        
        gramStyle = calculateGramMatrix(sfr);
        gramTransfer = calculateGramMatrix(tsfr);
        sLoss = regionalWeightFactor*mean((gramTransfer - gramStyle).^2,"all") / ((h*w*c)^2);
        
        loss = loss + (styleWeights(i)*sLoss);
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

function extracted_masks = extractMask(mask)
    
end
