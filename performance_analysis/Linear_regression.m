function [accuracy,TotalBatchTime]= Linear_regression()
disp('Linear Regression');
BatchTime=tic; 										% Start Time
load heart.txt;
data=heart;
[r c]=size(data);
randindex=randperm(r);
R = round(0.5*r);                       			% 50% training - 50% testing

train = data(randindex(1:R),:);
test = data(randindex(R+1:end),:);  				%randomized data


t=train(:,end);
x=train(:,1:end-1);
X=[ones(size(x,1),1) x]  ;          				%adding intercept term for bias

w=zeros(size(train,2),1);
eta=0.000001;                                       %learning rate
wlen = length(w);

for j=1:90
temp = (t-X*w);


for i=1:wlen
        Dw(i,1) = sum(temp.*X(:,i));
end

wold=w;
w = w + (eta)*Dw;                   				%compute w value

if (abs(wold-w)<0.0001)                 			%convergence of w value
    break;
end

end
 

 output=test*w;                            			%calculate output 09		
 
 
TotalBatchTime=toc(BatchTime); 						% End Time
 
out1=output;
out1(output<0)=1;
out1(output>0)=2;                       			%output labels assigned 

E=0.5*sum((out1-test(:,end)).^2); 					% error function calculated but not used

count=0;

for i=1:size(test,1)
    if test(i,end)==out1(i)
        count=count+1;
    end
end

accuracy=count/size(test,1)  ;          			%calculating accuracy
end
 