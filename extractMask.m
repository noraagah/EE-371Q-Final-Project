function [content_masks, style_masks] = extractMask(contentMask, styleMask, r)
% EXTRACTMASK returns the region guidance maps for the content and the
% style masks. The inputs are grayscale images of the content mask and the
% style mask.
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
        content_graylevels(i) = maj
        content_temp(content_temp == maj) = NaN;

%         maj = mode(style_temp(:));
%         style_graylevels(i) = maj;
%         style_temp(style_temp == maj) = NaN;
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