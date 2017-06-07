k=1:10;

for i=1:10
	fprintf('-----------------------------------Random Iteration : %d -------------------------------------- \n',i);
    [ Bayes_accuracy(i), Bayes_time(i)]=NaiveBayesClassifier();
   	[ svm_accuracy(i), svm_time(i)]=SVM();
   	[ Linear_reg_accuracy(i), Linear_reg_time(i)]=Linear_regression();
   	[ Logistic_reg_accuracy(i), Logistic_reg_time(i)]=Logistic_regression();
        end
figure,
plot(k,Bayes_accuracy,k,svm_accuracy,k,Linear_reg_accuracy,k,Logistic_reg_accuracy),legend('NaiveBayesClassfier','SVM','Linear regression','Logistic regression');
xlabel('Random iterations'),ylabel('accuracy'),title('Accuracy Analysis'),
figure,
plot(k,Bayes_time,k,svm_time,k,Linear_reg_time,k,Logistic_reg_time),axis([1 10 0 0.1]),xlabel('Random iterations'),ylabel('execution time in sec'),title('Performance Analysis'),legend('NaiveBayesClassfier','SVM','Linear regression','Logistic regression');
