%% Face Detection Code
% Code modified from: https://blogs.mathworks.com/pick/2014/03/14/detecting-faces-in-images/
%Function returns 1 if a face is present and a 0 if not and returns
%bounding boxes of faces

function [Face_Presence, bbox] = FaceDetection(image)
    faceDetector = vision.CascadeObjectDetector;
    shapeInserter = vision.ShapeInserter('BorderColor','Custom','CustomBorderColor',[0 255 255]);
    %Read in face image
    %I = imread('visionteam.jpg');
    %imshow(I);shg;
    %Get bounding boxes of faces
    bbox = step(faceDetector, image);
    % Draw boxes around detected faces and display results
    I_faces = step(shapeInserter, image, int32(bbox));
    imshow(I_faces), title('Detected faces');
    %Return true if faces are present
    Face_Presence = ~(isempty(bbox)); %Line added to original code

end