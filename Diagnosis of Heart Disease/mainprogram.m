
%----------------------------------------------SMO ALGORITHM---------------
clear;
clc;
load newheart.txt                                  % load the dataset
data = newheart;
%data=[3,0,0,1;0,3,3,-1;0,0,3,-1;3,3,0,1];

original_data = data;                               % copying original data
input = data(:,1:end-1);                           % training data
target = data(:,end);                                 %Y value of training data

C = 1;
epsi = 0.001;                                    %epsilon value 
alpha = 0;
b = 0;
%max_ITR = 6;                                        %maximum number of iterations 
counter = 1;
comp=0;
while length(find(alpha==0))>0              %fails when alpha vector contains only non-zero elements
    
    input = data(:,1:end-1);
    target = data(:,end);
    samples = length(target);
    alpha = zeros(samples,1);
    counter                                                                             % to display iteration number
   for i=1:samples
       
      % Error of sample i
       Ei = sum(alpha.*target.*kernel(input,input(i,:))) - target(i);  
    
     
      % It ensures that when the error is lower than the EPSILON
    %  if ((Ei*target(i) < -epsi) && (alpha(i)<C)) || ((Ei*target(i) > epsi) && (alpha(i) > 0))
     if (0<alpha(i)<C)     
          % Search over all j
          for j=1:samples
             
             % If they are both the same, skip it
             if j~=i
                 
                  % Error of sample j
                  Ej = sum(alpha.*target.*kernel(input,input(j,:))) - target(j); 
                  
                  alphaI = alpha(i);
                  alphaJ = alpha(j);

                  % Compute L and H boundary conditions
                  if target(i)==target(j)
                      L = max([0 alpha(i)+alpha(j)-C]);
                      H = min([C alpha(i)+alpha(j)]);
                  else
                      L = max([0 alpha(j)-alpha(i)]);
                      H = min([C C+alpha(j)-alpha(i)]);
                  end
                  
                 if H==L
                    
                     continue; % continue to next j
                  end

                  % Compute n(eta)
                  N = 2*kernel(input(j,:),input(i,:))-kernel(input(i,:),input(i,:))-kernel(input(j,:),input(j,:));
                  
                  if N >= 0
                  continue;
                  end

                  % Compute alpha 1 from KKT constraint:
                   alpha(j) = alpha(j) - (target(j)*(Ei-Ej))/N;
                                   
                  if alpha(j) > H
                      alpha(j) = H;
                  elseif alpha(j) < L
                      alpha(j) = L;
                  end
                  
                  if abs(alpha(j)-alphaJ) < epsi
                  continue;
                  end

                  % Alpha 1 is computed from alpha 2
                  alpha(i) = alpha(i) + target(i)*target(j)*(alphaJ-alpha(j));
           
                  % Compute b
                  b1 =Ei - target(i)*(alpha(i)-alphaI)*kernel(input(i,:),input(i,:))-target(j)*(alpha(j)-alphaJ)*kernel(input(i,:),input(j,:));
                  b2 =Ej - target(i)*(alpha(i)-alphaI)*kernel(input(i,:),input(j,:))-target(j)*(alpha(j)-alphaJ)*kernel(input(j,:),input(j,:));
                
                    
                  %constraints for b value
                  
                  if (0 < alpha(i)) && (alpha(i) < C)
                      b = b1;
                  elseif (0 < alpha(j)) && (alpha(j) < C)
                      b = b2;
                  else 
                      b=(b1+b2)/2;
                  end

             end
             
          end
          
      end 
   end
   
   counter=counter+1;
   
   % to eliminate data values whose respective lagrange multiplier's are still zero
   data=data((find(alpha~=0)),:);
   
end

% Calculate final W value
W = 0;
for i=1:samples
    W = W + alpha(i)*target(i)*input(i,:);
end

b = target(1) - input(1,:)*W';

Y=(original_data(:,1:end-1)*W')+b;

%to calculate accuracy of classification


for i=1:length(original_data)
    if Y(i)<0
        Y(i)=-1;
    else
        Y(i)=1;
    end  
end

disp('--------------CLASSIFIED VALUES------------------')
Y;

% classification test
for i=1:length(original_data)
    if Y(i)==original_data(i,size(original_data,2))
        comp=comp+1;
    end
end

% to compute the classification accuracy
Accuracy=comp/length(original_data);

