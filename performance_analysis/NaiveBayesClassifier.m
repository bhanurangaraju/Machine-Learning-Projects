function [classificationAccuracy,TotalBatchTime]=NaiveBayesClassifier()
disp('NaiveBayesClassifier');

BatchTime=tic;                          % Start Time
load heart.txt;                                                                

% for count
                                         % intervals or numBins
binCount=5;


    
        
        discretizedData = discrete(heart, binCount);                        %collecting discretized data
    
        randomIndices = randperm(270, 270);                                 % randomizing the indexes

        trainingDataSet = discretizedData(randomIndices(1:150), :);         %    1-150 trainingDataset
        testDataSet = discretizedData(randomIndices(150:270), :);           %    150-270 testdataset
    
        countSetosa = (trainingDataSet(:, end) == 1);                       % count of Setosa
        countNotSetosa = (trainingDataSet(:, end) ~= 1);                    % count of non-setosa

        probSetosa = sum(countSetosa)/length(trainingDataSet);              % probability of setosa
        probNotSetosa = sum(countNotSetosa)/length(trainingDataSet);        % probability of not Setosa
    
        conditionalProbabilitySet = ConditionalProbability(trainingDataSet, binCount);   % conditional probability function
    
        classificationAccuracy = TestClassifier(testDataSet, conditionalProbabilitySet, probSetosa, probNotSetosa);      % function to calculate accuracy
        
TotalBatchTime=toc(BatchTime);                                              % End Time

end
  function [classificationAccuracy] = TestClassifier (testDataSet, conditionalProbabilitySet, positiveProbability, negativeProbability)
    

    dataSize = size(testDataSet);
    
      
    classificationMatrix = zeros(dataSize(1), 2);
    classificationMatrix(:, 1) = testDataSet(:, 5); %copying the values of classes in test data to first column of classificationMatrix
    
 for rowNum = 1:dataSize(1)
        
        positiveClassification = 0;
        negativeClassification = 0;
        
        for columnNum = 1:dataSize(2) - 1
            
            positiveClassification = positiveClassification + (conditionalProbabilitySet(testDataSet(rowNum, columnNum), columnNum, 1))*positiveProbability;
            negativeClassification = negativeClassification + (conditionalProbabilitySet(testDataSet(rowNum, columnNum), columnNum, 2))*negativeProbability;
            
        end
        
        if positiveClassification > negativeClassification
            classificationMatrix(rowNum, 2) = 1;
        else
            classificationMatrix(rowNum, 2) = 2;
        end
        
 %compares original classes in first column of matrix with computed classes
%in second column of matrix and adds up number of successful cases
    
    classificationAccuracy = sum(classificationMatrix(:, 1) == classificationMatrix(:, 2))/dataSize(1) ;
 end
  end
    
   %function call for discretization

function[normalizedDataSet]=discrete(heartdata,numBins)		%Create an empty matrix, where we will load normalized data set
%normalizedDataSet = zeros(150,5);
															% Iterate on all columns of Data set
for i= 1:size(heartdata,2)-1        				 	    % we dont to normalize last column hence size -1
    x = heartdata(:,i);              						% iterate on columns
    binEdges = linspace(min(x), max(x) , numBins);    		% Get interval values by taking min and max values
    [bincount ,whichBin] = histc(x, binEdges);                 
    normalizedDataSet(:,i) = (whichBin);
end
% To create nrmalized data with labels
normalizedDataSet(:,end)= heartdata(:,end);
  
   
  
end

function [probabilitySet] = ConditionalProbability (dataSet, intervals)

    dataSize = size(dataSet);
    
    probabilitySet = zeros(intervals, dataSize(2) - 1, 2);
   
    
    positiveDataSet = dataSet(dataSet(:,5) == 1, :);
    negativeDataSet = dataSet(dataSet(:,5) ~= 1, :);
    
    for columnNum = 1:dataSize(2) - 1
    
        [n,x] = hist(positiveDataSet(:, columnNum), unique(positiveDataSet(:, columnNum)));
        p1 = n/sum(n);
        probabilitySet(x, columnNum, 1) = p1;
        
        [n,x] = hist(negativeDataSet(:, columnNum), unique(negativeDataSet(:, columnNum)));
        p2 = n/sum(n);
        probabilitySet(x, columnNum, 2) = p2;
        
        
    end
end
