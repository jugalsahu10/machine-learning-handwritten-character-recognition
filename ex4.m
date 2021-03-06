%% Neural Network Learning with Backpropagation

%% Initialization

clear; clc;
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 250;  % 200 hidden units
num_labels = 26;         % 26 labels, from 1 to 26, as there are 26 characters   
inputLayerSize = input_layer_size;
hiddenLayerSize = hidden_layer_size;
numberOfLabels = num_labels;
ch1 = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'];


%% Converting images into matrices and storing in X
X = zeros(1430,400);
for i = 1:26
    for j = 1:55
        path = strcat('training_set\',strcat(num2str(i),strcat(' (',strcat(num2str(j),').jpg'))));
        img_matrix = imread(path);
        img_matrix = img_matrix(:,:,1);
        img_matrix = img_matrix(:);
        img_matrix = 255 - img_matrix;
        X((i-1)*55+j,:)= img_matrix;
    end
end


%% y (b/w 1-26) is output value for each matrices
% In our data set 1st 55 are a's and next 55 are b's and so on
y = zeros(1430,1);
cnt = 1;
for i = 1:26
    for j = 1:55
        y((i-1)*55+j)=i;
    end
end
m = size(X, 1);


%% =========== Part 1: Visualizing Data =============

fprintf('Visualizing Data ...\n');

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));
title('Example of 100 Randomly Selected Characters from Input Training Set');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 6: Initializing Pameters ================
% Initializing Neural Network Parameters

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(inputLayerSize, hiddenLayerSize);
initial_Theta2 = randInitializeWeights(hiddenLayerSize, numberOfLabels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =================== Part 8: Training NN ===================
%  To train neural network, we are using "fmincg", which
%  is a function which works similarly to "fminunc". These
%  advanced optimizers are able to train cost functions efficiently as
%  long as we provide them with the gradient computations.

fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 100);

lambda = 0.5;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   inputLayerSize, ...
                                   hiddenLayerSize, ...
                                   numberOfLabels, X, y, lambda);

[neuralNetworkParameters, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from neuralNetworkParameters
Theta1 = reshape(neuralNetworkParameters(1:hiddenLayerSize * (inputLayerSize + 1)), ...
                 hiddenLayerSize, (inputLayerSize + 1));

Theta2 = reshape(neuralNetworkParameters((1 + (hiddenLayerSize * (inputLayerSize + 1))):end), ...
                 numberOfLabels, (hiddenLayerSize + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. We will use predict functino to predict output.

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


%% random test cases
disp('Test for 10 cases');
numberOfTrainingExamples = 1430;
rp = randperm(numberOfTrainingExamples);

for i = 1:10
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));
   

    predictions = predict(Theta1, Theta2, X(rp(i),:));
    cnt = predictions(1);
    fprintf('\nNeural Network Prediction: %c \n', ch1(cnt));
    
    % Pause
    fprintf('Program paused. Press enter to continue.\n');
    pause;
end