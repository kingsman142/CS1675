% Load the three images, including the pre-processing
panda = loadImage('panda.jpg');
cardinal = loadImage('cardinal.jpg');
pittsburgh = loadImage('pittsburgh.png');

% Run k-means with restarts with K = 4, iters = 10, restarts = 15
[pandaIds, pandaMeans, pandaSSD] = restarts(panda, 4, 10, 15);
[cardinalIds, cardinalMeans, cardinalSSD] = restarts(cardinal, 4, 10, 15);
[pittsburghIds, pittsburghMeans, pittsburghSSD] = restarts(pittsburgh, 4, 10, 15);

% After the optimal clusters are found, segment each image
newPanda = segmentImage(pandaIds, pandaMeans, panda);
newCardinal = segmentImage(cardinalIds, cardinalMeans, cardinal);
newPittsburgh = segmentImage(pittsburghIds, pittsburghMeans, pittsburgh);

% Reshape the images from H*W x 3 to 100x100x3, or HxWx3
newPanda = reshape(newPanda, [100, 100, 3]);
newCardinal = reshape(newCardinal, [100, 100, 3]);
newPittsburgh = reshape(newPittsburgh, [100, 100, 3]);

% Show each image side by side
imshow([newPanda, newCardinal, newPittsburgh]);

% Pre-process the image
function im = loadImage(filename)
    im = imread(filename); % read image
    im = double(im); % convert all values to doubles
    im = imresize(im, [100 100]); % size the original image to 100x100
    H = 100;
    W = 100;
    im = reshape(im, H*W, 3); % reshape the image to a H*W x 3 matrix
end

% Segment the image based on the computed means
function im = segmentImage(ids, means, im)
    for i = 1:size(ids,1)
       id = ids(i);
       meanVal = means(id,:);
       im(i,:) = meanVal;
    end
    im = uint8(im); % convert back from double to uint8
end