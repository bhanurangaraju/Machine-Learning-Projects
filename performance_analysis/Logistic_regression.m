function [Accuracy,TotalBatchTime]=Logistic_regression()
disp('Logistic Regression');

BatchTime=tic; 										% Start Time
load heart.txt;
data=heart;
[r, c]=size(data);
col1 = ones(r,1);
X=[col1 data];										% Extend by 1 to account for the bias

p=0.5;
max_itr=5; 											% to bound the number of iterations
count=0;
randindex=randperm(r);
N = round(p*r);

newclass(data(:,end) == 1) = 0;
newclass(data(:,end) == 2) = 1;
train = X(randindex(1:N),:);
trainlabels = newclass(randindex(1:N));
test = X(randindex(N+1:r),:);
testlabels = newclass(randindex(N+1:r));

[tr1 tc1]=size(train);
w=zeros(tc1,1);										% initialize w
Xt=train;
y=trainlabels;
ybar=mean(trainlabels);
w(1)=log(ybar/(1-ybar));
s=zeros(1,N);
z=zeros(N,1);
eps=0.001;											%learning rate


													%Algorithm test for convergence
while(count<max_itr)
    count=count+1;
    eta=Xt*w+w(1);
    mue=1./(1+exp(-eta));
    s=mue.*(1-mue);
    z=eta+(y'-mue)./s;
    S=diag(s);
    wold=w;
    w=pinv(Xt'*S*Xt)* Xt' *S *z; 
    w(isnan(w))=0;
    if (abs(w-wold)<=eps)              				%convergence condition
        break;
    end
end

ltest=length(testlabels);							%Test


% Compute the output for each test data using w
for i=1:ltest,
out(i)=test(i,:)*w;
end
				
TotalBatchTime=toc(BatchTime); 						% End Time

% Transform output in 0 1 labels
out1=out;
out1(out<0)=0;
out1(out>0)=1;

% compute accuracy
Accuracy = 1 - sum(abs(testlabels - out1))/ltest;


end