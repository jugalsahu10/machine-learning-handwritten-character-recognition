function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
                               
% reshaping Theta1 and Theta2 from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
en = hidden_layer_size*(input_layer_size+1)+(num_labels*(hidden_layer_size+1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):en), num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% We need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

        %% forword propogation and total cost calculation
           a = zeros(m,input_layer_size+hidden_layer_size+num_labels);
            for i = 1:m
                % a1, a2, a3 are for three layers
                a1 = zeros(input_layer_size,1);
                a2 = zeros(hidden_layer_size,1);
                a3 = zeros(num_labels,1);
                a1 = X(i,:)';
                a(i,1:input_layer_size) = a1';
                a1 = [1;a1];
                a2 = sigmoid(Theta1*a1);
                a(i,1+input_layer_size:input_layer_size+hidden_layer_size) = a2';
                a2 = [1 a2']';
                a3 = sigmoid(Theta2*a2);
                a(i,input_layer_size+hidden_layer_size+1:input_layer_size+hidden_layer_size+num_labels) = a3';
                yy = zeros(num_labels,1);
                yy(y(i)) = 1;
                
                % Summing cost
                J = J + sum(-yy.*log(a3)-(1-yy).*log(1-a3));
            end
                J = J/m;
               
            %% Regularization for cost
            [x yt] = size(Theta1_grad);
            JJ = 0;
            for i=1:x
                for j=2:yt
                    JJ = JJ + (Theta1(i,j)^2);
                end
            end
            [x yt] = size(Theta2_grad);
            for i=1:x
                for j=2:yt
                    JJ = JJ + (Theta2(i,j)^2);
                end
            end
            J = J + (lambda*JJ)/(2*m);
        
            %% back propogation and calculating gradient of thetas
            for i = 1:m
                    a1 = a(i,1:input_layer_size)';
                    a2 = a(i,input_layer_size+1:input_layer_size+hidden_layer_size)';
                    a3 = a(i,input_layer_size+hidden_layer_size+1:end)';
                    yy = zeros(num_labels,1);
                    yy(y(i)) = 1;
                    delta = a3 - yy;
                    Theta2_grad = Theta2_grad + delta*([1;a2]');
                    delta = Theta2'*delta;
                    delta = delta(2:end);
                    delta = delta.*(a2.*(1-a2));
                    Theta1_grad = Theta1_grad + delta*([1;a1]');
            end
            
            Theta1_grad = Theta1_grad/m;
            Theta2_grad = Theta2_grad/m;
            [x y] = size(Theta1_grad);
            
            %% Regularization
            for i=1:x
                for j=2:y
                    Theta1_grad(i,j) = Theta1_grad(i,j)+ (lambda*Theta1(i,j))/m;
                end
            end
            [x y] = size(Theta2_grad);
            for i=1:x
                for j=2:y
                    Theta2_grad(i,j) = Theta2_grad(i,j)+ (lambda*Theta2(i,j))/m;
                end
            end

    grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
